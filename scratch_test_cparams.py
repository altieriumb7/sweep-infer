import ctypes
import os

os.add_dll_directory(r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama")
lib = ctypes.CDLL(r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\libllama.dll")

# We can call llama_context_default_params and see the returned struct or byte buffer
# In C, struct is returned by value or pointer depending on ABI. On x64 Windows, structs larger than 8 bytes are returned via pointer in RCX.
class DummyBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 512)]

lib.llama_context_default_params.restype = DummyBuffer
cparams = lib.llama_context_default_params()
print("Context default params obtained, length 512 buffer filled.")
