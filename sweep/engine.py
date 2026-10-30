"""
SWEEP High-Level Python Engine API
Author: Umberto Altieri
License: MIT
"""

import ctypes
import os
import sys
import time
import glob
import numpy as np
from typing import List, Dict, Any, Generator, Tuple, Optional

# Locate llama.cpp shared libraries
def find_llama_library() -> Tuple[Optional[ctypes.CDLL], Optional[ctypes.CDLL]]:
    possible_dirs = [
        "/workspace/llama.cpp/build-cuda/bin",
        "/workspace/llama.cpp/build-cpu/bin",
        "/usr/local/lib",
        r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama",
        os.path.join(os.path.dirname(__file__), "..", "build", "bin"),
        os.path.join(os.path.dirname(__file__), "..", "lib"),
    ]
    
    libllama = None
    libggml = None
    
    lib_names = ["libllama.so", "llama.dll", "libllama.dll", "libllama.dylib"]
    ggml_names = ["libggml.so", "ggml.dll", "libggml.dylib"]

    for d in possible_dirs:
        if not os.path.exists(d):
            continue
        for l_name in lib_names:
            p_llama = os.path.join(d, l_name)
            if os.path.exists(p_llama):
                try:
                    if hasattr(os, 'add_dll_directory') and sys.platform == 'win32':
                        os.add_dll_directory(d)
                    # Check ggml
                    for g_name in ggml_names:
                        p_ggml = os.path.join(d, g_name)
                        if os.path.exists(p_ggml):
                            libggml = ctypes.CDLL(p_ggml)
                            if hasattr(libggml, 'ggml_backend_load_all_from_path'):
                                libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
                                libggml.ggml_backend_load_all_from_path(d.encode('utf-8'))
                            break
                    libllama = ctypes.CDLL(p_llama)
                    libllama.llama_backend_init()
                    return libllama, libggml
                except Exception:
                    continue
    return None, None

class ModelParamsBuf(ctypes.Structure):
    _fields_ = [('data', ctypes.c_uint8 * 512)]

class CtxBuf(ctypes.Structure):
    _fields_ = [('ints', ctypes.c_int32 * 64)]

class llama_batch(ctypes.Structure):
    _fields_ = [
        ('n_tokens', ctypes.c_int32),
        ('token', ctypes.POINTER(ctypes.c_int32)),
        ('embd', ctypes.POINTER(ctypes.c_float)),
        ('pos', ctypes.POINTER(ctypes.c_int32)),
        ('n_seq_id', ctypes.POINTER(ctypes.c_int32)),
        ('seq_id', ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ('logits', ctypes.POINTER(ctypes.c_int8)),
    ]

# Statistical bigram prior
class BigramRouter:
    def __init__(self, prompt_tokens: List[int]):
        self.transitions: Dict[int, Dict[int, int]] = {}
        for i in range(len(prompt_tokens) - 1):
            self.observe(prompt_tokens[i], prompt_tokens[i+1])

    def observe(self, t1: int, t2: int):
        if t1 not in self.transitions:
            self.transitions[t1] = {}
        self.transitions[t1][t2] = self.transitions[t1].get(t2, 0) + 1

    def query(self, t: int, min_dominance: float = 0.55, min_count: int = 2) -> int:
        if t in self.transitions:
            m = self.transitions[t]
            best_tok, count = max(m.items(), key=lambda x: x[1])
            tot = sum(m.values())
            if count >= min_count and (count / tot) >= min_dominance:
                return best_tok
        return -1


class SweepEngine:
    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        n_threads: int = 8,
        n_ctx: int = 2048,
        wavefront_k: int = 2,
        entropy_threshold: float = 0.55,
        margin_threshold: float = 0.60,
    ):
        self.model_path = os.path.abspath(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.wavefront_k = wavefront_k
        self.entropy_threshold = entropy_threshold
        self.margin_threshold = margin_threshold

        self.libllama, self.libggml = find_llama_library()
        if not self.libllama:
            raise RuntimeError("Could not find or load libllama dynamic library.")

        # Setup ctypes prototypes
        self._init_c_bindings()

        # Load Model
        mparams = self.libllama.llama_model_default_params()
        if self.n_gpu_layers > 0:
            ctypes.cast(ctypes.byref(mparams), ctypes.POINTER(ctypes.c_int32))[4] = self.n_gpu_layers

        self.model = self.libllama.llama_model_load_from_file(self.model_path.encode('utf-8'), mparams)
        if not self.model:
            raise RuntimeError(f"Failed to load model from path: {self.model_path}")

        self.vocab = self.libllama.llama_model_get_vocab(self.model)
        self.vocab_size = self.libllama.llama_vocab_n_tokens(self.vocab)
        self.eos_tokens = {128001, 128008, 128009, 151643, 151644, 151645}

        # Try loading native AVX-2 Top-K entropy kernel
        self.native_topk = self._load_native_topk()

    def _init_c_bindings(self):
        ll = self.libllama
        ll.llama_model_default_params.restype = ModelParamsBuf
        ll.llama_context_default_params.restype = CtxBuf
        ll.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParamsBuf]
        ll.llama_model_load_from_file.restype = ctypes.c_void_p
        ll.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        ll.llama_model_get_vocab.restype = ctypes.c_void_p
        ll.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        ll.llama_vocab_n_tokens.restype = ctypes.c_int32
        ll.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxBuf]
        ll.llama_init_from_model.restype = ctypes.c_void_p
        ll.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        ll.llama_batch_init.restype = llama_batch
        ll.llama_batch_free.argtypes = [llama_batch]
        ll.llama_batch_free.restype = None
        ll.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
        ll.llama_decode.restype = ctypes.c_int32
        ll.llama_get_logits.argtypes = [ctypes.c_void_p]
        ll.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
        ll.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        ll.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
        ll.llama_get_memory.argtypes = [ctypes.c_void_p]
        ll.llama_get_memory.restype = ctypes.c_void_p
        ll.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        ll.llama_memory_seq_rm.restype = ctypes.c_bool
        ll.llama_tokenize.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
        ]
        ll.llama_tokenize.restype = ctypes.c_int32
        ll.llama_token_to_piece.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
        ]
        ll.llama_token_to_piece.restype = ctypes.c_int32
        ll.llama_free.argtypes = [ctypes.c_void_p]
        ll.llama_model_free.argtypes = [ctypes.c_void_p]

    def _load_native_topk(self):
        try:
            from synaptic_fold.kernels.native_router import compute_topk_entropy_avx2
            return compute_topk_entropy_avx2
        except Exception:
            return None

    def tokenize(self, text: str) -> List[int]:
        b = text.encode('utf-8')
        buf = (ctypes.c_int32 * (len(b) + 256))()
        n = self.libllama.llama_tokenize(self.vocab, b, len(b), buf, len(buf), False, True)
        return [buf[i] for i in range(n)]

    def token_to_str(self, token: int) -> str:
        buf = ctypes.create_string_buffer(128)
        rlen = self.libllama.llama_token_to_piece(self.vocab, token, buf, 128, 0, False)
        if rlen > 0:
            return buf.raw[:rlen].decode('utf-8', errors='replace')
        return ""

    def _compute_entropy(self, l_ptr, k=5):
        if self.native_topk:
            return self.native_topk(l_ptr, self.vocab_size, k)
        # Fallback NumPy
        logits = np.ctypeslib.as_array(l_ptr, shape=(self.vocab_size,))
        top_idx = np.argpartition(logits, -k)[-k:]
        top_idx = top_idx[np.argsort(-logits[top_idx])]
        top_vals = logits[top_idx]
        exp_v = np.exp(top_vals - np.max(top_vals))
        probs = exp_v / np.sum(exp_v)
        H = -np.sum(probs * np.log2(probs + 1e-12))
        margin = probs[0] - (probs[1] if k > 1 else 0.0)
        return top_idx, probs, H, margin

    def generate(self, prompt: str, max_tokens: int = 128) -> Generator[str, None, None]:
        prompt_tokens = self.tokenize(prompt)
        cparams = self.libllama.llama_context_default_params()
        cparams.ints[0] = self.n_ctx
        cparams.ints[1] = 2048 # n_batch
        cparams.ints[7] = self.n_threads
        cparams.ints[8] = self.n_threads

        ctx = self.libllama.llama_init_from_model(self.model, cparams)
        mem = self.libllama.llama_get_memory(ctx)
        router = BigramRouter(prompt_tokens)

        # Prefill in chunks of 512
        CHUNK = 512
        for offset in range(0, len(prompt_tokens), CHUNK):
            chunk = prompt_tokens[offset:offset+CHUNK]
            batch_p = self.libllama.llama_batch_init(len(chunk) + 16, 0, 1)
            for i, tok in enumerate(chunk):
                batch_p.token[i] = tok
                batch_p.pos[i] = offset + i
                batch_p.n_seq_id[i] = 1
                batch_p.seq_id[i][0] = 0
                batch_p.logits[i] = 1 if (offset + i == len(prompt_tokens) - 1) else 0
            batch_p.n_tokens = len(chunk)
            self.libllama.llama_decode(ctx, batch_p)
            self.libllama.llama_batch_free(batch_p)

        l_ptr = self.libllama.llama_get_logits(ctx)
        idx, p, H_local, margin = self._compute_entropy(l_ptr, 5)
        next_token = int(idx[0])
        history = list(prompt_tokens)

        generated = 0
        curr_pos = len(prompt_tokens)
        batch = self.libllama.llama_batch_init(16, 0, 1)

        while generated < max_tokens:
            if next_token in self.eos_tokens:
                break

            gate_open = (H_local < self.entropy_threshold and margin >= self.margin_threshold) and (generated + 2 <= max_tokens)
            draft_tok = router.query(next_token, min_dominance=0.55, min_count=2) if gate_open else -1

            if draft_tok != -1 and draft_tok not in self.eos_tokens:
                # Wavefront Speculation K=2
                batch.token[0] = next_token
                batch.pos[0] = curr_pos
                batch.n_seq_id[0] = 1
                batch.seq_id[0][0] = 0
                batch.logits[0] = 1

                batch.token[1] = draft_tok
                batch.pos[1] = curr_pos + 1
                batch.n_seq_id[1] = 1
                batch.seq_id[1][0] = 0
                batch.logits[1] = 1
                batch.n_tokens = 2

                ret = self.libllama.llama_decode(ctx, batch)
                if ret != 0: break

                l0_ptr = self.libllama.llama_get_logits_ith(ctx, 0)
                idx0, p0, H0, m0 = self._compute_entropy(l0_ptr, 5)
                pred_tok = int(idx0[0])

                yield self.token_to_str(next_token)
                generated += 1
                router.observe(next_token, pred_tok)
                history.append(next_token)

                if pred_tok == draft_tok:
                    yield self.token_to_str(draft_tok)
                    generated += 1
                    l1_ptr = self.libllama.llama_get_logits_ith(ctx, 1)
                    idx1, p1, H1, m1 = self._compute_entropy(l1_ptr, 5)
                    next_after = int(idx1[0])
                    router.observe(draft_tok, next_after)
                    history.append(draft_tok)
                    curr_pos += 2
                    next_token = next_after
                    H_local = H1; margin = m1
                else:
                    self.libllama.llama_memory_seq_rm(mem, 0, curr_pos + 1, -1)
                    curr_pos += 1
                    next_token = pred_tok
                    H_local = H0; margin = m0
            else:
                # Non-speculative single decode
                batch.token[0] = next_token
                batch.pos[0] = curr_pos
                batch.n_seq_id[0] = 1
                batch.seq_id[0][0] = 0
                batch.logits[0] = 1
                batch.n_tokens = 1

                ret = self.libllama.llama_decode(ctx, batch)
                if ret != 0: break

                yield self.token_to_str(next_token)
                generated += 1
                curr_pos += 1

                l_ptr = self.libllama.llama_get_logits(ctx)
                idx, p, H_local, margin = self._compute_entropy(l_ptr, 5)
                router.observe(next_token, int(idx[0]))
                history.append(next_token)
                next_token = int(idx[0])

        self.libllama.llama_batch_free(batch)
        self.libllama.llama_free(ctx)

    def close(self):
        if self.model:
            self.libllama.llama_model_free(self.model)
            self.model = None
