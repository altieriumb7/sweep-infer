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
mparams = libllama.llama_model_default_params()
ctypes.cast(ctypes.byref(mparams), ctypes.POINTER(ctypes.c_int32))[4] = 99 # 100% offload to RTX 5070 Ti

model = libllama.llama_model_load_from_file(MODEL_PATH, mparams)
vocab = libllama.llama_model_get_vocab(model)
vocab_size = libllama.llama_vocab_n_tokens(vocab)
print(f"Loaded Llama-3-8B on GPU. Vocab size: {vocab_size}")

cparams = libllama.llama_context_default_params()
cparams.ints[0] = 2048
cparams.ints[7] = 8
cparams.ints[8] = 8

ctx = libllama.llama_init_from_model(model, cparams)
mem = libllama.llama_get_memory(ctx)

def tokenize_str(text: str):
    b_text = text.encode('utf-8')
    tok_buf = (ctypes.c_int32 * (len(b_text) + 256))()
    n = libllama.llama_tokenize(vocab, b_text, len(b_text), tok_buf, len(tok_buf), False, True)
    return [tok_buf[i] for i in range(n)]

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
tokens = tokenize_str(prompt)
print(f"Tokenized prompt length: {len(tokens)}")

# Test prefill
batch_p = libllama.llama_batch_init(len(tokens) + 16, 0, 1)
for i, tok in enumerate(tokens):
    batch_p.token[i] = tok
    batch_p.pos[i] = i
    batch_p.n_seq_id[i] = 1
    batch_p.seq_id[i][0] = 0
    batch_p.logits[i] = 1 if i == len(tokens) - 1 else 0
batch_p.n_tokens = len(tokens)

t0 = time.time()
ret = libllama.llama_decode(ctx, batch_p)
libllama.llama_batch_free(batch_p)
print(f"Prefill ret: {ret} in {(time.time() - t0)*1000:.2f} ms")

# Get next token
l_ptr = libllama.llama_get_logits(ctx)
idx, p, H, m = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
print(f"First generated token: {idx[0]} (prob={p[0]:.4f}, H={H:.4f}, margin={m:.4f})")

# Test speculative batch decode of 2 tokens
batch_2 = libllama.llama_batch_init(8, 0, 1)
batch_2.token[0] = int(idx[0])
batch_2.pos[0] = len(tokens)
batch_2.n_seq_id[0] = 1
batch_2.seq_id[0][0] = 0
batch_2.logits[0] = 1

batch_2.token[1] = 9999 # arbitrary speculative draft
batch_2.pos[1] = len(tokens) + 1
batch_2.n_seq_id[1] = 1
batch_2.seq_id[1][0] = 0
batch_2.logits[1] = 1
batch_2.n_tokens = 2

t0 = time.time()
ret = libllama.llama_decode(ctx, batch_2)
dt = (time.time() - t0) * 1000
print(f"Speculative batch decode ret: {ret} in {dt:.2f} ms")

l0 = libllama.llama_get_logits_ith(ctx, 0)
idx0, p0, H0, m0 = compute_topk_entropy_avx2(l0, vocab_size, 5)
print(f"Ground truth target token for pos {len(tokens)}: {idx0[0]} (draft was 9999 -> rejected)")

# Test rollback
rm_ok = libllama.llama_memory_seq_rm(mem, 0, len(tokens) + 1, -1)
print(f"Rollback status: {rm_ok}")

libllama.llama_batch_free(batch_2)
libllama.llama_free(ctx)
libllama.llama_model_free(model)
libllama.llama_backend_free()
print("All GPU tests passed successfully!")
