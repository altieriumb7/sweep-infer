import ctypes
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ollama_lib_dir = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(ollama_lib_dir)

libggml = ctypes.CDLL(os.path.join(ollama_lib_dir, "ggml.dll"))
libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
libggml.ggml_backend_load_all_from_path(ollama_lib_dir.encode('utf-8'))

libllama = ctypes.CDLL(os.path.join(ollama_lib_dir, "libllama.dll"))
libllama.llama_backend_init()

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

mparams = libllama.llama_model_default_params()
cparams = libllama.llama_context_default_params()
cparams.ints[0] = 2048  # n_ctx
cparams.ints[7] = 8     # n_threads = 8
cparams.ints[8] = 8     # n_threads_batch = 8

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

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

libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_free.restype = None

libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

print("[1] Loading 8B model into memory...")
t0 = time.time()
model_ptr = libllama.llama_model_load_from_file(MODEL_BLOB, mparams)
print(f"[1] Loaded in {time.time() - t0:.2f} s")

print("[2] Initializing context...")
ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
print(f"[2] Context initialized: {hex(ctx_ptr)}")

# Allocate batch of 16 tokens
batch = libllama.llama_batch_init(16, 0, 1)

# Prompt token: 128000 (<|begin_of_text|>)
batch.token[0] = 128000
batch.pos[0] = 0
batch.n_seq_id[0] = 1
batch.seq_id[0][0] = 0
batch.logits[0] = 1
batch.n_tokens = 1

print("[3] Testing 1-token decode (Sequential GEMV baseline)...")
t0 = time.time()
ret = libllama.llama_decode(ctx_ptr, batch)
t_single = time.time() - t0
print(f"[3] Single token decode finished with code {ret} in {t_single*1000:.2f} ms ({1.0/t_single:.2f} tok/s)!")

# Test 4-token batched decode in 1 sweep
print("[4] Testing 4-token parallel batched decode (Picard Wavefront sweep)...")
for i in range(4):
    batch.token[i] = 1000 + i
    batch.pos[i] = 1 + i
    batch.n_seq_id[i] = 1
    batch.seq_id[i][0] = 0
    batch.logits[i] = 1
batch.n_tokens = 4

t0 = time.time()
ret4 = libllama.llama_decode(ctx_ptr, batch)
t_batch = time.time() - t0
speed_batch = 4.0 / t_batch
latency_per_tok = (t_batch * 1000.0) / 4.0

print(f"[4] 4-token batched sweep finished in {t_batch*1000:.2f} ms total!")
print(f"[4] -> Latenza media per token nel fronte d'onda: {latency_per_tok:.2f} ms/token")
print(f"[4] -> Velocita di generazione parallela: {speed_batch:.2f} tok/s!")

libllama.llama_batch_free(batch)
libllama.llama_free(ctx_ptr)
libllama.llama_model_free(model_ptr)
libllama.llama_backend_free()
print("[5] Completato con successo.")
