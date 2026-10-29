import ctypes
import os
import sys
import time
import numpy as np

# Setup DLL paths
OLLAMA_LIB_DIR = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(OLLAMA_LIB_DIR)

libggml = ctypes.CDLL(os.path.join(OLLAMA_LIB_DIR, "ggml.dll"))
libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
libggml.ggml_backend_load_all_from_path(OLLAMA_LIB_DIR.encode('utf-8'))

libllama = ctypes.CDLL(os.path.join(OLLAMA_LIB_DIR, "libllama.dll"))
libllama.llama_backend_init()

CURRENT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from synaptic_fold.kernels.native_router import (
    evaluate_speculative_tree_avx2,
    compute_column_potentials_avx2,
    compute_topk_entropy_avx2,
    is_native_avx2_available,
    is_native_tree_avx2_available,
    is_native_topk_avx2_available
)

# Ctypes Prototypes
class ModelParamsBuf(ctypes.Structure):
    _fields_ = [('data', ctypes.c_uint8 * 128)]

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

libllama.llama_model_default_params.restype = ModelParamsBuf
libllama.llama_context_default_params.restype = CtxBuf
libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParamsBuf]
libllama.llama_model_load_from_file.restype = ctypes.c_void_p
libllama.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
libllama.llama_model_get_vocab.restype = ctypes.c_void_p
libllama.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxBuf]
libllama.llama_init_from_model.restype = ctypes.c_void_p
libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_batch_init.restype = llama_batch
libllama.llama_batch_free.argtypes = [llama_batch]
libllama.llama_batch_free.restype = None
libllama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
libllama.llama_decode.restype = ctypes.c_int32
libllama.llama_get_logits.argtypes = [ctypes.c_void_p]
libllama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
libllama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
libllama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
libllama.llama_get_memory.argtypes = [ctypes.c_void_p]
libllama.llama_get_memory.restype = ctypes.c_void_p
libllama.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_memory_seq_rm.restype = ctypes.c_bool
libllama.llama_tokenize.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
]
libllama.llama_tokenize.restype = ctypes.c_int32
libllama.llama_token_to_piece.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
]
libllama.llama_token_to_piece.restype = ctypes.c_int32
libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_free.restype = None
libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

# CRC32 Router DLL
ROUTER_DLL_PATH = r"C:\Users\Umberto\synaptic-fold-v2\synaptic_fold\kernels\synaptic_ngram_router_avx2.dll"
librouter = ctypes.CDLL(ROUTER_DLL_PATH)
librouter.synaptic_ngram_router_init.argtypes = []
librouter.synaptic_ngram_router_init.restype = None
librouter.synaptic_ngram_router_insert.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
librouter.synaptic_ngram_router_insert.restype = None
librouter.synaptic_ngram_router_query.argtypes = [ctypes.c_int32, ctypes.c_int32]
librouter.synaptic_ngram_router_query.restype = ctypes.c_int32
librouter.synaptic_ngram_router_seed.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
librouter.synaptic_ngram_router_seed.restype = None

def tokenize(vocab, text):
    b = text.encode('utf-8')
    buf = (ctypes.c_int32 * (len(b) + 64))()
    n = libllama.llama_tokenize(vocab, b, len(b), buf, len(buf), False, True)
    return [buf[i] for i in range(n)]

def to_str(vocab, tok):
    buf = ctypes.create_string_buffer(64)
    rlen = libllama.llama_token_to_piece(vocab, tok, buf, 64, 0, False)
    return buf.raw[:rlen].decode('utf-8', errors='replace') if rlen > 0 else ""

def test_run():
    print("Test Zero-Penalty Speculation on 25 tokens...")
    mparams = libllama.llama_model_default_params()
    cparams = libllama.llama_context_default_params()
    cparams.ints[0] = 1024
    cparams.ints[7] = 8
    cparams.ints[8] = 8

    model_ptr = libllama.llama_model_load_from_file(MODEL_BLOB, mparams)
    vocab_ptr = libllama.llama_model_get_vocab(model_ptr)
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    mem_handle = libllama.llama_get_memory(ctx_ptr)

    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nQual e' la capitale d'Italia?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    tokens = tokenize(vocab_ptr, prompt)

    # Prefill
    batch = libllama.llama_batch_init(len(tokens) + 16, 0, 1)
    for i, t in enumerate(tokens):
        batch.token[i] = t
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if i == len(tokens) - 1 else 0
    batch.n_tokens = len(tokens)
    libllama.llama_decode(ctx_ptr, batch)
    libllama.llama_batch_free(batch)

    # Router init
    librouter.synaptic_ngram_router_init()
    tok_arr = (ctypes.c_int32 * len(tokens))(*tokens)
    librouter.synaptic_ngram_router_seed(tok_arr, len(tokens))

    batch = libllama.llama_batch_init(8, 0, 1)
    curr_pos = len(tokens)
    history = list(tokens)
    generated = []

    last_logits = libllama.llama_get_logits(ctx_ptr)
    idx, probs, H_local, margin = compute_topk_entropy_avx2(last_logits, 128256, 5)
    next_token = int(idx[0])

    N_STEPS = 25
    decodes_count = 0
    draft_proposed = 0
    draft_accepted = 0

    t0 = time.time()
    while len(generated) < N_STEPS:
        if next_token in (128001, 128009):
            break

        gate_open = (H_local < 0.60 and margin >= 0.50) and (len(generated) + 2 <= N_STEPS)
        draft_tok = -1
        if gate_open and len(history) >= 1:
            prev = history[-1]
            draft_tok = librouter.synaptic_ngram_router_query(prev, next_token)
            if draft_tok in (128001, 128009):
                draft_tok = -1

        # For testing, let's intentionally test speculation on the first step with a deliberate dummy draft to test divergence rollback
        if len(generated) == 0:
            draft_tok = 9999 # Deliberate wrong draft to test rollback

        if draft_tok != -1:
            draft_proposed += 1
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

            decodes_count += 1
            ret = libllama.llama_decode(ctx_ptr, batch)
            assert ret == 0

            l0_ptr = libllama.llama_get_logits_ith(ctx_ptr, 0)
            idx0, probs0, H0, m0 = compute_topk_entropy_avx2(l0_ptr, 128256, 5)
            pred_tok = int(idx0[0])

            generated.append(next_token)
            librouter.synaptic_ngram_router_insert(history[-1], next_token, pred_tok)
            history.append(next_token)

            if pred_tok == draft_tok:
                # Accepted
                draft_accepted += 1
                generated.append(draft_tok)
                l1_ptr = libllama.llama_get_logits_ith(ctx_ptr, 1)
                idx1, probs1, H1, m1 = compute_topk_entropy_avx2(l1_ptr, 128256, 5)
                next_after = int(idx1[0])

                librouter.synaptic_ngram_router_insert(history[-1], draft_tok, next_after)
                history.append(draft_tok)

                curr_pos += 2
                next_token = next_after
                H_local = H1
                margin = m1
                print(f"  [SPEC ACCEPT] pos {curr_pos-2}->{curr_pos}: '{to_str(vocab_ptr, next_token)}' & '{to_str(vocab_ptr, draft_tok)}'")
            else:
                # Rejected: ZERO PENALTY! Rollback draft_tok at curr_pos + 1
                libllama.llama_memory_seq_rm(mem_handle, 0, curr_pos + 1, -1)
                curr_pos += 1
                next_token = pred_tok
                H_local = H0
                margin = m0
                print(f"  [SPEC REJECT] pos {curr_pos-1}: committed '{to_str(vocab_ptr, generated[-1])}', next is '{to_str(vocab_ptr, next_token)}' (0 penalty)")
        else:
            # Single decode
            batch.token[0] = next_token
            batch.pos[0] = curr_pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1
            batch.n_tokens = 1

            decodes_count += 1
            ret = libllama.llama_decode(ctx_ptr, batch)
            assert ret == 0

            generated.append(next_token)
            curr_pos += 1

            l_ptr = libllama.llama_get_logits(ctx_ptr)
            idx, probs, H_local, margin = compute_topk_entropy_avx2(l_ptr, 128256, 5)
            pred_tok = int(idx[0])

            librouter.synaptic_ngram_router_insert(history[-1], next_token, pred_tok)
            history.append(next_token)
            next_token = pred_tok

    dt = time.time() - t0
    gen_text = "".join([to_str(vocab_ptr, t) for t in generated])
    print(f"\nGenerazione Completata: {len(generated)} token in {dt:.2f}s ({len(generated)/dt:.2f} tok/s)")
    print(f"Decodes eseguiti: {decodes_count} per {len(generated)} token")
    print(f"Draft proposti: {draft_proposed}, accettati: {draft_accepted}")
    print(f"Testo generato:\n{gen_text}\n")

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)
    libllama.llama_model_free(ctypes.c_void_p(model_ptr))
    libllama.llama_backend_free()

if __name__ == '__main__':
    test_run()
