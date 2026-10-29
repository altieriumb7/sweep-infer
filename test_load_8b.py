import ctypes
import os
import time

ollama_lib_dir = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(ollama_lib_dir)

libggml = ctypes.CDLL(os.path.join(ollama_lib_dir, "ggml.dll"))
libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
libggml.ggml_backend_load_all_from_path(ollama_lib_dir.encode('utf-8'))

libllama = ctypes.CDLL(os.path.join(ollama_lib_dir, "libllama.dll"))
libllama.llama_backend_init()

class ModelParamsBuf(ctypes.Structure):
    _fields_ = [('data', ctypes.c_uint8 * 128)]

libllama.llama_model_default_params.restype = ModelParamsBuf
mparams = libllama.llama_model_default_params()

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParamsBuf]
libllama.llama_model_load_from_file.restype = ctypes.c_void_p

t0 = time.time()
print("Loading 8B model into process RAM...")
model_ptr = libllama.llama_model_load_from_file(MODEL_BLOB, mparams)
print(f"Model loaded in {time.time() - t0:.2f} s! Model ptr: {hex(model_ptr) if model_ptr else 'NULL'}")

if model_ptr:
    libllama.llama_model_free.argtypes = [ctypes.c_void_p]
    libllama.llama_model_free(model_ptr)
    print("Model freed.")

libllama.llama_backend_free()
print("Backend freed.")
