import ctypes
import os
import sys
import time
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

OLLAMA_LIB_DIR = r"C:\Users\Umberto\AppData\Local\Programs\Ollama\lib\ollama"
os.add_dll_directory(OLLAMA_LIB_DIR)

libggml = ctypes.CDLL(os.path.join(OLLAMA_LIB_DIR, "ggml.dll"))
libggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
libggml.ggml_backend_load_all_from_path(OLLAMA_LIB_DIR.encode('utf-8'))

libllama = ctypes.CDLL(os.path.join(OLLAMA_LIB_DIR, "libllama.dll"))
libllama.llama_backend_init()

CURRENT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from synaptic_fold.kernels.native_router import (
    evaluate_speculative_tree_avx2,
    compute_column_potentials_avx2,
    is_native_avx2_available,
    is_native_tree_avx2_available
)
from synaptic_fold.tree_speculator import EntropyGuidedTreeSpeculator

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
libllama.llama_tokenize.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
]
libllama.llama_tokenize.restype = ctypes.c_int32
libllama.llama_token_to_piece.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
]
libllama.llama_token_to_piece.restype = ctypes.c_int32
libllama.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_memory_seq_rm.restype = ctypes.c_bool
libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_free.restype = None
libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

MODEL_BLOB = r"C:\Users\Umberto\.ollama\models\blobs\sha256-6a0746a1ec1aef3e7ec53868f220ff6e389f6f8ef87a01d77c96807de94ca2aa".encode('utf-8')

mparams = libllama.llama_model_default_params()
cparams = libllama.llama_context_default_params()
cparams.ints[0] = 2048
cparams.ints[7] = 8
cparams.ints[8] = 8

print("Loading model...")
model_ptr = libllama.llama_model_load_from_file(MODEL_BLOB, mparams)
vocab_ptr = libllama.llama_model_get_vocab(model_ptr)
ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
print("Context initialized.")

# Test llama_memory_seq_rm
ok = libllama.llama_memory_seq_rm(ctx_ptr, 0, 10, -1)
print("llama_memory_seq_rm result:", ok)

libllama.llama_free(ctx_ptr)
libllama.llama_model_free(ctypes.c_void_p(model_ptr))
libllama.llama_backend_free()
print("Cleaned up successfully!")
