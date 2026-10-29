import ctypes
import os

LIB_DIR = "/workspace/llama.cpp/build-cpu/bin"
libllama = ctypes.CDLL(os.path.join(LIB_DIR, "libllama.so"))
libllama.llama_backend_init()

class ModelParamsBuf(ctypes.Structure):
    _fields_ = [('data', ctypes.c_uint8 * 128)]

libllama.llama_model_default_params.restype = ModelParamsBuf
libllama.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParamsBuf]
libllama.llama_model_load_from_file.restype = ctypes.c_void_p
libllama.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
libllama.llama_model_get_vocab.restype = ctypes.c_void_p

libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

if hasattr(libllama, 'llama_vocab_n_tokens'):
    libllama.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
    libllama.llama_vocab_n_tokens.restype = ctypes.c_int32
    print("Found llama_vocab_n_tokens")
elif hasattr(libllama, 'llama_n_vocab'):
    libllama.llama_n_vocab.argtypes = [ctypes.c_void_p]
    libllama.llama_n_vocab.restype = ctypes.c_int32
    print("Found llama_n_vocab")

mparams = libllama.llama_model_default_params()
for mpath in ["/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"]:
    if os.path.exists(mpath):
        m = libllama.llama_model_load_from_file(mpath.encode('utf-8'), mparams)
        v = libllama.llama_model_get_vocab(m)
        nv = libllama.llama_vocab_n_tokens(v) if hasattr(libllama, 'llama_vocab_n_tokens') else -1
        print(f"Model: {os.path.basename(mpath)} -> n_vocab: {nv}")
        libllama.llama_model_free(ctypes.c_void_p(m))
