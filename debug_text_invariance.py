import ctypes
import os
import sys
import time
import numpy as np

PROJECT_ROOT = '/workspace/sweep'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LIB_DIR = '/workspace/llama.cpp/build-cuda/bin'
libggml = ctypes.CDLL(os.path.join(LIB_DIR, 'libggml.so'))
libllama = ctypes.CDLL(os.path.join(LIB_DIR, 'libllama.so'))
libllama.llama_backend_init()

from synaptic_fold.kernels.native_router import (
    compute_topk_entropy_avx2,
    init_ngram_router_avx2,
    seed_ngram_router_avx2,
    insert_ngram_router_avx2,
    query_ngram_router_avx2,
)
from synaptic_fold.tree_speculator import SyntacticGrammarPrior
from benchmarks.benchmark_datasets import GSM8K_SAMPLES, HUMANEVAL_SAMPLES, BOOLQ_SAMPLES

# Silence verbose llama.cpp logging
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def quiet_log(level, text, user_data): pass
c_quiet_log = LOG_CALLBACK(quiet_log)
libllama.llama_log_set(c_quiet_log, None)

class ModelParamsBuf(ctypes.Structure): _fields_ = [('data', ctypes.c_uint8 * 128)]
class CtxBuf(ctypes.Structure): _fields_ = [('ints', ctypes.c_int32 * 64)]

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
libllama.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
libllama.llama_vocab_n_tokens.restype = ctypes.c_int32
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

MODEL_PATH = '/workspace/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'.encode('utf-8')
EOS_TOKENS = {128001, 128008, 128009}

mparams = libllama.llama_model_default_params()
ctypes.cast(ctypes.byref(mparams), ctypes.POINTER(ctypes.c_int32))[4] = 99
model_ptr = libllama.llama_model_load_from_file(MODEL_PATH, mparams)
vocab_ptr = libllama.llama_model_get_vocab(model_ptr)
vocab_size = libllama.llama_vocab_n_tokens(vocab_ptr)

def tokenize_str(text: str):
    b_text = text.encode('utf-8')
    tok_buf = (ctypes.c_int32 * (len(b_text) + 256))()
    n = libllama.llama_tokenize(vocab_ptr, b_text, len(b_text), tok_buf, len(tok_buf), False, True)
    return [tok_buf[i] for i in range(n)]

def token_to_str(token: int) -> str:
    buf = ctypes.create_string_buffer(128)
    rlen = libllama.llama_token_to_piece(vocab_ptr, token, buf, 128, 0, False)
    if rlen > 0:
        return buf.raw[:rlen].decode('utf-8', errors='replace')
    return ""

def format_chat_prompt(user_text: str) -> str:
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

class DominantBigramSpeculator:
    def __init__(self, prompt_tokens):
        self.transitions = {}
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

def decode_prompt_chunks(ctx_ptr, prompt_tokens):
    CHUNK = 512
    for offset in range(0, len(prompt_tokens), CHUNK):
        chunk = prompt_tokens[offset:offset+CHUNK]
        batch_p = libllama.llama_batch_init(len(chunk) + 16, 0, 1)
        for i, tok in enumerate(chunk):
            batch_p.token[i] = tok
            batch_p.pos[i] = offset + i
            batch_p.n_seq_id[i] = 1
            batch_p.seq_id[i][0] = 0
            batch_p.logits[i] = 1 if (offset + i == len(prompt_tokens) - 1) else 0
        batch_p.n_tokens = len(chunk)
        ret = libllama.llama_decode(ctx_ptr, batch_p)
        libllama.llama_batch_free(batch_p)
        assert ret == 0

def run_baseline_sequential(prompt_tokens, n_predict: int):
    cparams = libllama.llama_context_default_params()
    cparams.ints[0] = 2048
    cparams.ints[1] = 2048
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    batch = libllama.llama_batch_init(16, 0, 1)
    decode_prompt_chunks(ctx_ptr, prompt_tokens)

    l_ptr = libllama.llama_get_logits(ctx_ptr)
    idx, _, _, _ = compute_topk_entropy_avx2(l_ptr, vocab_size, 1)
    next_token = int(idx[0])

    generated = []
    curr_pos = len(prompt_tokens)
    t0 = time.perf_counter()

    while len(generated) < n_predict:
        if next_token in EOS_TOKENS:
            break
        batch.token[0] = next_token
        batch.pos[0] = curr_pos
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1
        batch.n_tokens = 1
        ret = libllama.llama_decode(ctx_ptr, batch)
        if ret != 0: break
        generated.append(next_token)
        curr_pos += 1

        l_ptr = libllama.llama_get_logits(ctx_ptr)
        idx, _, _, _ = compute_topk_entropy_avx2(l_ptr, vocab_size, 1)
        next_token = int(idx[0])

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)
    return generated

def run_sweep_wavefront(prompt_tokens, n_predict: int, grammar_prior):
    cparams = libllama.llama_context_default_params()
    cparams.ints[0] = 2048
    cparams.ints[1] = 2048
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    mem_handle = libllama.llama_get_memory(ctx_ptr)
    batch = libllama.llama_batch_init(16, 0, 1)

    init_ngram_router_avx2()
    seed_ngram_router_avx2(prompt_tokens)
    bigram_speculator = DominantBigramSpeculator(prompt_tokens)
    decode_prompt_chunks(ctx_ptr, prompt_tokens)

    l_ptr = libllama.llama_get_logits(ctx_ptr)
    idx, p, H_local, margin = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
    next_token = int(idx[0])
    history = list(prompt_tokens)

    generated = []
    curr_pos = len(prompt_tokens)

    while len(generated) < n_predict:
        if next_token in EOS_TOKENS: break

        gate_open = (H_local < 0.55 and margin >= 0.60) and (len(generated) + 2 <= n_predict)
        draft_tok = -1

        if gate_open and len(history) >= 1:
            prev_tok = history[-1]
            draft_tok = query_ngram_router_avx2(prev_tok, next_token)
            if draft_tok == -1:
                draft_tok = bigram_speculator.query(next_token, min_dominance=0.55, min_count=2)
            if draft_tok in EOS_TOKENS: draft_tok = -1
            if draft_tok != -1:
                if not grammar_prior.is_valid_transition(token_to_str(next_token), token_to_str(draft_tok)):
                    draft_tok = -1

        if draft_tok != -1:
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
            ret = libllama.llama_decode(ctx_ptr, batch)
            if ret != 0: break

            l0_ptr = libllama.llama_get_logits_ith(ctx_ptr, 0)
            idx0, p0, H0, m0 = compute_topk_entropy_avx2(l0_ptr, vocab_size, 5)
            pred_tok = int(idx0[0])

            generated.append(next_token)
            insert_ngram_router_avx2(history[-1], next_token, pred_tok)
            bigram_speculator.observe(next_token, pred_tok)
            history.append(next_token)

            if pred_tok == draft_tok:
                generated.append(draft_tok)
                l1_ptr = libllama.llama_get_logits_ith(ctx_ptr, 1)
                idx1, p1, H1, m1 = compute_topk_entropy_avx2(l1_ptr, vocab_size, 5)
                next_after_draft = int(idx1[0])
                insert_ngram_router_avx2(history[-1], draft_tok, next_after_draft)
                bigram_speculator.observe(draft_tok, next_after_draft)
                history.append(draft_tok)
                curr_pos += 2
                next_token = next_after_draft
                p = p1; H_local = H1; margin = m1
            else:
                libllama.llama_memory_seq_rm(mem_handle, 0, curr_pos + 1, -1)
                curr_pos += 1
                next_token = pred_tok
                p = p0; H_local = H0; margin = m0
        else:
            batch.token[0] = next_token
            batch.pos[0] = curr_pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1
            batch.n_tokens = 1
            ret = libllama.llama_decode(ctx_ptr, batch)
            if ret != 0: break

            generated.append(next_token)
            curr_pos += 1

            l_ptr = libllama.llama_get_logits(ctx_ptr)
            idx, p, H_local, margin = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
            insert_ngram_router_avx2(history[-1], next_token, int(idx[0]))
            bigram_speculator.observe(next_token, int(idx[0]))
            history.append(next_token)
            next_token = int(idx[0])

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)
    return generated

grammar_prior = SyntacticGrammarPrior()

def compare(name, prompt_text, max_toks=48):
    fmt = format_chat_prompt(prompt_text)
    toks = tokenize_str(fmt)
    b = run_baseline_sequential(toks, max_toks)
    s = run_sweep_wavefront(toks, max_toks, grammar_prior)
    b_txt = "".join([token_to_str(t) for t in b])
    s_txt = "".join([token_to_str(t) for t in s])

    print("=" * 75)
    print(f"CASE: {name}")
    print(f"Tokens Generated: Baseline={len(b)}, SWEEP={len(s)}")
    print(f"Tokens Identical: {b == s}")
    print("--- BASELINE OUTPUT ---")
    print(b_txt.strip())
    print("--- SWEEP OUTPUT ---")
    print(s_txt.strip())
    print("=" * 75)

compare("GSM8K #1 (Clips Problem)", GSM8K_SAMPLES[0]["question"])
compare("GSM8K #3 (Betty Wallet Problem)", GSM8K_SAMPLES[2]["question"])
compare("HumanEval #1 (has_close_elements)", HUMANEVAL_SAMPLES[0]["prompt"])
compare("BoolQ #1 (Phantom of the Opera)", f"Passage: {BOOLQ_SAMPLES[0]['passage']}\nQuestion: {BOOLQ_SAMPLES[0]['question']}\nAnswer:")

libllama.llama_model_free(model_ptr)
libllama.llama_backend_free()
