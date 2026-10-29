import ctypes
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ollama_lib_dir = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(ollama_lib_dir)

libggml = ctypes.CDLL(os.path.join(ollama_lib_dir, "ggml.dll"))
libllama = ctypes.CDLL(os.path.join(ollama_lib_dir, "libllama.dll"))

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

print("[0] Loading all GGML hardware compute backends...")
libggml.ggml_backend_load_all()

print("[1] Initializing llama backend...")
libllama.llama_backend_init()

class ModelParamsBuf(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 128)]

class CtxParamsBuf(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 512)]

libllama.llama_model_default_params.restype = ModelParamsBuf
libllama.llama_context_default_params.restype = CtxParamsBuf

mparams = libllama.llama_model_default_params()
cparams = libllama.llama_context_default_params()

libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParamsBuf]
libllama.llama_model_load_from_file.restype = ctypes.c_void_p

libllama.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxParamsBuf]
libllama.llama_init_from_model.restype = ctypes.c_void_p

libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_free.restype = None

libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

print("[2] Loading 8B model into process RAM...")
t0 = time.time()
model_ptr = libllama.llama_model_load_from_file(MODEL_BLOB, mparams)
print(f"[2] Model loaded in {time.time() - t0:.2f} s! Model ptr: {hex(model_ptr) if model_ptr else 'NULL'}")

if model_ptr:
    print("[3] Creating context...")
    t0 = time.time()
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    print(f"[3] Context created in {time.time() - t0:.2f} s! Ctx ptr: {hex(ctx_ptr) if ctx_ptr else 'NULL'}")

    print("[4] Cleaning up...")
    if ctx_ptr:
        libllama.llama_free(ctx_ptr)
    libllama.llama_model_free(model_ptr)

libllama.llama_backend_free()
print("[5] Done successfully!")
