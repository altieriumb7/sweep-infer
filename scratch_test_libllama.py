import ctypes
import os

dll_path = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\libllama.dll"
# Add Ollama dir to DLL search path so dependent DLLs (ggml.dll, ggml-cpu-zen4.dll, etc.) are resolved
os.add_dll_directory(r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama")

lib = ctypes.CDLL(dll_path)
print("libllama.dll loaded successfully!")

# Check common functions
funcs = [
    "llama_backend_init",
    "llama_backend_free",
    "llama_model_default_params",
    "llama_context_default_params",
    "llama_model_load_from_file",
    "llama_init_from_model",
    "llama_free",
    "llama_model_free",
    "llama_decode",
    "llama_batch_init",
    "llama_batch_free",
    "llama_get_logits_ith",
    "llama_sampler_init_greedy",
    "llama_sampler_sample",
]

for f in funcs:
    has = hasattr(lib, f)
    print(f"Function {f:30}: {'FOUND' if has else 'MISSING'}")
