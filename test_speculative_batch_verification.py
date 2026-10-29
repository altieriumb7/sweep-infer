import ctypes
import os
import sys
import time
import numpy as np

OLLAMA_LIB_DIR = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(OLLAMA_LIB_DIR)

libggml = ctypes.CDLL(os.path.join(OLLAMA_LIB_DIR, "ggml.dll"))
libggml.ggml_backend_load_all_from_path(OLLAMA_LIB_DIR.encode('utf-8'))

libllama = ctypes.CDLL(os.path.join(OLLAMA_LIB_DIR, "libllama.dll"))
libllama.llama_backend_init()

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
libllama.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxBuf]
libllama.llama_init_from_model.restype = ctypes.c_void_p
libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_batch_init.restype = llama_batch
libllama.llama_batch_free.argtypes = [llama_batch]
libllama.llama_batch_free.restype = None
libllama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
libllama.llama_decode.restype = ctypes.c_int32
libllama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
libllama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
libllama.llama_get_memory.argtypes = [ctypes.c_void_p]
libllama.llama_get_memory.restype = ctypes.c_void_p
libllama.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_memory_seq_rm.restype = ctypes.c_bool
libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.argtypes = [ctypes.c_void_p]

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

m = libllama.llama_model_load_from_file(MODEL_BLOB, libllama.llama_model_default_params())
cparams = libllama.llama_context_default_params()
cparams.ints[0] = 512
cparams.ints[7] = 8
cparams.ints[8] = 8
ctx = libllama.llama_init_from_model(m, cparams)
mem = libllama.llama_get_memory(ctx)

# Test decoding 2 tokens with logits=1 on both
batch = libllama.llama_batch_init(8, 0, 1)
batch.token[0] = 128000
batch.pos[0] = 0
batch.n_seq_id[0] = 1
batch.seq_id[0][0] = 0
batch.logits[0] = 1

batch.token[1] = 1000
batch.pos[1] = 1
batch.n_seq_id[1] = 1
batch.seq_id[1][0] = 0
batch.logits[1] = 1
batch.n_tokens = 2

t0 = time.time()
ret = libllama.llama_decode(ctx, batch)
dt = time.time() - t0
print(f"Decode ret={ret} in {dt*1000:.2f} ms")

l0 = libllama.llama_get_logits_ith(ctx, 0)
l1 = libllama.llama_get_logits_ith(ctx, 1)
print("l0 valid:", bool(l0), "l1 valid:", bool(l1))

# Rollback position 1
ok = libllama.llama_memory_seq_rm(mem, 0, 1, -1)
print("Rollback position 1:", ok)

libllama.llama_batch_free(batch)
libllama.llama_free(ctx)
libllama.llama_model_free(ctypes.c_void_p(m))
libllama.llama_backend_free()
print("Success!")
