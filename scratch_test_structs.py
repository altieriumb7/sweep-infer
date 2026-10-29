import ctypes
import os

os.add_dll_directory(r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama")
lib = ctypes.CDLL(r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\libllama.dll")

class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("use_mmap", ctypes.c_bool),
        ("use_mlock", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
    ]

lib.llama_model_default_params.restype = llama_model_params
mparams = lib.llama_model_default_params()
print("Model default params obtained successfully!")
print("  n_gpu_layers:", mparams.n_gpu_layers)
print("  use_mmap:", mparams.use_mmap)
