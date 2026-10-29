import ctypes
import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ensure Ollama dependencies (ggml, libcommon, openmp) are discoverable
os.add_dll_directory(r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama")
libllama_path = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama\libllama.dll"
lib = ctypes.CDLL(libllama_path)

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

# Initialize backend
lib.llama_backend_init()
print("[1] Backend initialized.")

# Model params
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
mparams.use_mmap = True

print(f"[2] Loading 8B model from file: {MODEL_BLOB.decode()}...")
t0 = time.time()
lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
lib.llama_model_load_from_file.restype = ctypes.c_void_p
model_ptr = lib.llama_model_load_from_file(MODEL_BLOB, mparams)
t_load = time.time() - t0

if not model_ptr:
    print("[ERRORE] Impossibile caricare il modello.")
    sys.exit(1)

print(f"[2] Modello 8B caricato in-process in {t_load:.2f} s! Pointer: {hex(model_ptr)}")

# Context params
class llama_context_params(ctypes.Structure):
    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int32),
        ("pooling_type", ctypes.c_int32),
        ("attention_type", ctypes.c_int32),
        ("flash_attn_type", ctypes.c_int32),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ctypes.c_void_p),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int32),
        ("type_v", ctypes.c_int32),
        ("logits_all", ctypes.c_bool),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("flash_attn", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("abort_callback", ctypes.c_void_p),
        ("abort_callback_data", ctypes.c_void_p),
    ]

# Try creating context
lib.llama_init_from_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p] # use pointer to default params
lib.llama_init_from_model.restype = ctypes.c_void_p

# Call default params
class RawParams(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 512)]

lib.llama_context_default_params.restype = RawParams
raw_cparams = lib.llama_context_default_params()

# Modify n_ctx in raw buffer if known or pass pointer
ctx_ptr = lib.llama_init_from_model(model_ptr, ctypes.byref(raw_cparams))
print(f"[3] Contesto creato in-process! Pointer: {hex(ctx_ptr) if ctx_ptr else 'NULL'}")

# Clean up
if ctx_ptr:
    lib.llama_free(ctx_ptr)
if model_ptr:
    lib.llama_model_free(model_ptr)
lib.llama_backend_free()
print("[4] Cleanup completato.")
