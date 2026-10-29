#!/usr/bin/env python3
# ==============================================================================
# SWEEP: Definitive Academic Benchmark Suite on NVIDIA RTX 5070 Ti (Blackwell)
# Target Architecture: Meta-Llama-3-8B-Instruct (4.58 GiB, Q4_K_M, 100% VRAM)
# Author: Umberto Altieri
#
# Suite Components:
#   1. Multi-Domain Academic Invariance & Speedup (GSM8K, HumanEval, BoolQ)
#   2. Context-Length Horizon Scaling (N_ctx = 128, 512, 1024, 2048, 4000)
#   3. Power & Energy Efficiency Telemetry (nvidia-smi polling, Joules/token)
# ==============================================================================

import ctypes
import os
import sys
import time
import json
import threading
import subprocess
import numpy as np
from typing import List, Tuple, Dict, Any

# Ensure stdout is UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = '/workspace/sweep'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LIB_DIR = '/workspace/llama.cpp/build-cuda/bin'
libggml = ctypes.CDLL(os.path.join(LIB_DIR, 'libggml.so'))
libllama = ctypes.CDLL(os.path.join(LIB_DIR, 'libllama.so'))
libllama.llama_backend_init()

from synaptic_fold.kernels.native_router import (
    compute_topk_entropy_avx2,
    evaluate_speculative_tree_avx2,
    compute_column_potentials_avx2,
    init_ngram_router_avx2,
    seed_ngram_router_avx2,
    insert_ngram_router_avx2,
    query_ngram_router_avx2,
)
from synaptic_fold.tree_speculator import SyntacticGrammarPrior
from benchmarks.benchmark_datasets import GSM8K_SAMPLES, HUMANEVAL_SAMPLES, BOOLQ_SAMPLES

# Silence verbose llama.cpp logging
LOG_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def quiet_log(level, text, user_data): pass
c_quiet_log = LOG_CALLBACK(quiet_log)
libllama.llama_log_set(c_quiet_log, None)

# Ctypes Structs
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
libllama.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
libllama.llama_vocab_n_tokens.restype = ctypes.c_int32
libllama.llama_init_from_model.argtypes = [ctypes.c_void_p, CtxBuf]
libllama.llama_init_from_model.restype = ctypes.c_void_p
libllama.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_batch_init.restype = llama_batch
libllama.llama_batch_free.argtypes = [llama_batch]
libllama.llama_batch_free.restype = None
libllama.llama_decode.argtypes = [ctypes.c_void_p, llama_batch]
libllama.llama_decode.restype = ctypes.c_int32
libllama.llama_get_logits.argtypes = [ctypes.c_void_p]
libllama.llama_get_logits.restype = ctypes.POINTER(ctypes.c_float)
libllama.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
libllama.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
libllama.llama_get_memory.argtypes = [ctypes.c_void_p]
libllama.llama_get_memory.restype = ctypes.c_void_p
libllama.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
libllama.llama_memory_seq_rm.restype = ctypes.c_bool
libllama.llama_tokenize.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
]
libllama.llama_tokenize.restype = ctypes.c_int32
libllama.llama_token_to_piece.argtypes = [
    ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
]
libllama.llama_token_to_piece.restype = ctypes.c_int32
libllama.llama_free.argtypes = [ctypes.c_void_p]
libllama.llama_free.restype = None
libllama.llama_model_free.argtypes = [ctypes.c_void_p]
libllama.llama_model_free.restype = None

# Model Loading Helper
MODEL_PATH = '/workspace/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf'.encode('utf-8')
EOS_TOKENS = {128001, 128008, 128009}

print("=" * 85)
print("  SWEEP: DEFINITIVE ACADEMIC BENCHMARK SUITE ON NVIDIA GEFORCE RTX 5070 Ti")
print("  Target: Meta-Llama-3-8B-Instruct (4.58 GiB, Q4_K_M, 100% Offload VRAM GDDR7)")
print("  Author: Umberto Altieri")
print("=" * 85)

mparams = libllama.llama_model_default_params()
ctypes.cast(ctypes.byref(mparams), ctypes.POINTER(ctypes.c_int32))[4] = 99 # 100% GPU Offload

t0_load = time.time()
model_ptr = libllama.llama_model_load_from_file(MODEL_PATH, mparams)
if not model_ptr:
    print("[FATAL] Could not load model into GPU memory.")
    sys.exit(1)
vocab_ptr = libllama.llama_model_get_vocab(model_ptr)
vocab_size = libllama.llama_vocab_n_tokens(vocab_ptr)
t_load = time.time() - t0_load
print(f"[INIT] Model loaded in {t_load:.2f} s | Vocab size: {vocab_size} | Architecture: Blackwell SM 12.0")

def tokenize_str(text: str) -> List[int]:
    b_text = text.encode('utf-8')
    tok_buf = (ctypes.c_int32 * (len(b_text) + 256))()
    n = libllama.llama_tokenize(vocab_ptr, b_text, len(b_text), tok_buf, len(tok_buf), False, True)
    return [tok_buf[i] for i in range(n)]

def token_to_str(token: int) -> str:
    buf = ctypes.create_string_buffer(128)
    rlen = libllama.llama_token_to_piece(vocab_ptr, token, buf, 128, 0, False)
    if rlen > 0:
        return buf.raw[:rlen].decode('utf-8', errors='replace')
    return ""

def format_chat_prompt(user_text: str) -> str:
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

# Bigram Statistical Router
class DominantBigramSpeculator:
    def __init__(self, prompt_tokens: List[int]):
        self.transitions: Dict[int, Dict[int, int]] = {}
        for i in range(len(prompt_tokens) - 1):
            self.observe(prompt_tokens[i], prompt_tokens[i+1])

    def observe(self, t1: int, t2: int):
        if t1 not in self.transitions:
            self.transitions[t1] = {}
        self.transitions[t1][t2] = self.transitions[t1].get(t2, 0) + 1

    def query(self, t: int, min_dominance: float = 0.55, min_count: int = 2) -> int:
        if t in self.transitions:
            m = self.transitions[t]
            best_tok, count = max(m.items(), key=lambda x: x[1])
            tot = sum(m.values())
            if count >= min_count and (count / tot) >= min_dominance:
                return best_tok
        return -1

def decode_prompt_chunks(ctx_ptr, prompt_tokens: List[int]):
    """Decodes prompt in chunks to strictly adhere to n_batch limits."""
    CHUNK = 512
    for offset in range(0, len(prompt_tokens), CHUNK):
        chunk = prompt_tokens[offset:offset+CHUNK]
        batch_p = libllama.llama_batch_init(len(chunk) + 16, 0, 1)
        for i, tok in enumerate(chunk):
            batch_p.token[i] = tok
            batch_p.pos[i] = offset + i
            batch_p.n_seq_id[i] = 1
            batch_p.seq_id[i][0] = 0
            batch_p.logits[i] = 1 if (offset + i == len(prompt_tokens) - 1) else 0
        batch_p.n_tokens = len(chunk)
        ret = libllama.llama_decode(ctx_ptr, batch_p)
        libllama.llama_batch_free(batch_p)
        assert ret == 0, f"Prompt prefill chunk failed: ret={ret}"

# 1. Baseline Sequential Greedy Runner
def run_baseline_sequential(prompt_tokens: List[int], n_predict: int, n_ctx: int = 4096) -> Tuple[List[int], float, float]:
    cparams = libllama.llama_context_default_params()
    cparams.ints[0] = n_ctx
    cparams.ints[1] = 2048 # n_batch
    cparams.ints[7] = 8
    cparams.ints[8] = 8
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    batch = libllama.llama_batch_init(16, 0, 1)

    # Prefill in chunks
    decode_prompt_chunks(ctx_ptr, prompt_tokens)

    l_ptr = libllama.llama_get_logits(ctx_ptr)
    idx, _, _, _ = compute_topk_entropy_avx2(l_ptr, vocab_size, 1)
    next_token = int(idx[0])

    generated = []
    curr_pos = len(prompt_tokens)
    t0 = time.perf_counter()

    while len(generated) < n_predict:
        if next_token in EOS_TOKENS:
            break
        batch.token[0] = next_token
        batch.pos[0] = curr_pos
        batch.n_seq_id[0] = 1
        batch.seq_id[0][0] = 0
        batch.logits[0] = 1
        batch.n_tokens = 1

        ret = libllama.llama_decode(ctx_ptr, batch)
        if ret != 0:
            break
        generated.append(next_token)
        curr_pos += 1

        l_ptr = libllama.llama_get_logits(ctx_ptr)
        idx, _, _, _ = compute_topk_entropy_avx2(l_ptr, vocab_size, 1)
        next_token = int(idx[0])

    elapsed = time.perf_counter() - t0
    tps = len(generated) / elapsed if elapsed > 0 else 0.0

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)
    return generated, elapsed, tps

# 2. SWEEP Speculative Wavefront Runner (Zero Rollback Penalty)
def run_sweep_wavefront(
    prompt_tokens: List[int], n_predict: int,
    grammar_prior: SyntacticGrammarPrior,
    entropy_gate_thresh: float = 0.55, margin_thresh: float = 0.60,
    n_ctx: int = 4096
) -> Dict[str, Any]:
    cparams = libllama.llama_context_default_params()
    cparams.ints[0] = n_ctx
    cparams.ints[1] = 2048 # n_batch
    cparams.ints[7] = 8
    cparams.ints[8] = 8
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    mem_handle = libllama.llama_get_memory(ctx_ptr)
    batch = libllama.llama_batch_init(16, 0, 1)

    init_ngram_router_avx2()
    seed_ngram_router_avx2(prompt_tokens)
    bigram_speculator = DominantBigramSpeculator(prompt_tokens)

    # Prefill in chunks
    decode_prompt_chunks(ctx_ptr, prompt_tokens)

    l_ptr = libllama.llama_get_logits(ctx_ptr)
    idx, p, H_local, margin = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
    next_token = int(idx[0])
    history = list(prompt_tokens)

    generated = []
    curr_pos = len(prompt_tokens)
    total_proposed = 0
    total_accepted = 0
    sweeps_count = 0

    t0 = time.perf_counter()

    while len(generated) < n_predict:
        if next_token in EOS_TOKENS:
            break

        gate_open = (H_local < entropy_gate_thresh and margin >= margin_thresh) and (len(generated) + 2 <= n_predict)
        draft_tok = -1

        if gate_open and len(history) >= 1:
            prev_tok = history[-1]
            draft_tok = query_ngram_router_avx2(prev_tok, next_token)
            if draft_tok == -1:
                draft_tok = bigram_speculator.query(next_token, min_dominance=0.55, min_count=2)

            if draft_tok in EOS_TOKENS:
                draft_tok = -1

            if draft_tok != -1:
                next_str = token_to_str(next_token)
                draft_str = token_to_str(draft_tok)
                if not grammar_prior.is_valid_transition(next_str, draft_str):
                    draft_tok = -1

        if draft_tok != -1:
            # SPECULATIVE WAVEFRONT K=2
            total_proposed += 1
            sweeps_count += 1

            batch.token[0] = next_token
            batch.pos[0] = curr_pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1

            batch.token[1] = draft_tok
            batch.pos[1] = curr_pos + 1
            batch.n_seq_id[1] = 1
            batch.seq_id[1][0] = 0
            batch.logits[1] = 1

            batch.n_tokens = 2
            ret = libllama.llama_decode(ctx_ptr, batch)
            if ret != 0:
                break

            l0_ptr = libllama.llama_get_logits_ith(ctx_ptr, 0)
            idx0, p0, H0, m0 = compute_topk_entropy_avx2(l0_ptr, vocab_size, 5)
            pred_tok = int(idx0[0])

            generated.append(next_token)
            insert_ngram_router_avx2(history[-1], next_token, pred_tok)
            bigram_speculator.observe(next_token, pred_tok)
            history.append(next_token)

            if pred_tok == draft_tok:
                # ACCEPTED: 2 tokens verified in 1 sweep
                total_accepted += 1
                generated.append(draft_tok)

                l1_ptr = libllama.llama_get_logits_ith(ctx_ptr, 1)
                idx1, p1, H1, m1 = compute_topk_entropy_avx2(l1_ptr, vocab_size, 5)
                next_after_draft = int(idx1[0])

                insert_ngram_router_avx2(history[-1], draft_tok, next_after_draft)
                bigram_speculator.observe(draft_tok, next_after_draft)
                history.append(draft_tok)

                curr_pos += 2
                next_token = next_after_draft
                p = p1
                H_local = H1
                margin = m1
            else:
                # REJECTED: Zero Penalty Rollback
                libllama.llama_memory_seq_rm(mem_handle, 0, curr_pos + 1, -1)
                curr_pos += 1
                next_token = pred_tok
                p = p0
                H_local = H0
                margin = m0
        else:
            # NON-SPECULATIVE SINGLE DECODE
            sweeps_count += 1
            batch.token[0] = next_token
            batch.pos[0] = curr_pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1
            batch.n_tokens = 1

            ret = libllama.llama_decode(ctx_ptr, batch)
            if ret != 0:
                break

            generated.append(next_token)
            curr_pos += 1

            l_ptr = libllama.llama_get_logits(ctx_ptr)
            idx, p, H_local, margin = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
            insert_ngram_router_avx2(history[-1], next_token, int(idx[0]))
            bigram_speculator.observe(next_token, int(idx[0]))
            history.append(next_token)
            next_token = int(idx[0])

    elapsed = time.perf_counter() - t0
    tps = len(generated) / elapsed if elapsed > 0 else 0.0
    alpha = (total_accepted / total_proposed) if total_proposed > 0 else 0.0

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)

    return {
        'generated_tokens': generated,
        'elapsed_sec': elapsed,
        'tokens_per_sec': tps,
        'sweeps_count': sweeps_count,
        'total_proposed': total_proposed,
        'total_accepted': total_accepted,
        'alpha': alpha
    }

# ==============================================================================
# BENCHMARK PART 1: Multi-Domain Academic Generalization & Bit-Level Invariance
# ==============================================================================
grammar_prior = SyntacticGrammarPrior()

def run_academic_domain(domain_name: str, samples: list, prompt_key: str, n_predict: int = 48):
    print(f"\n--- Domain Benchmark: {domain_name} ({len(samples)} prompts, max_tokens={n_predict}) ---")
    base_tps_list = []
    sweep_tps_list = []
    alpha_list = []
    invariance_count = 0
    total_tokens_base = 0
    total_tokens_sweep = 0

    for i, item in enumerate(samples):
        if domain_name == "BoolQ":
            raw_prompt = f"Passage: {item['passage']}\nQuestion: {item['question']}\nAnswer:"
        else:
            raw_prompt = item[prompt_key]
        formatted = format_chat_prompt(raw_prompt)
        prompt_tokens = tokenize_str(formatted)

        base_tokens, base_time, base_tps = run_baseline_sequential(prompt_tokens, n_predict)
        sweep_res = run_sweep_wavefront(prompt_tokens, n_predict, grammar_prior)

        # Invariance check
        sweep_tokens = sweep_res['generated_tokens']
        is_identical = (base_tokens == sweep_tokens)
        if is_identical:
            invariance_count += 1

        base_tps_list.append(base_tps)
        sweep_tps_list.append(sweep_res['tokens_per_sec'])
        if sweep_res['total_proposed'] > 0:
            alpha_list.append(sweep_res['alpha'])
        total_tokens_base += len(base_tokens)
        total_tokens_sweep += len(sweep_tokens)

        if (i + 1) % 5 == 0 or (i + 1) == len(samples):
            print(f"  [{i+1:>2}/{len(samples)}] Baseline: {base_tps:.1f} tok/s | SWEEP: {sweep_res['tokens_per_sec']:.1f} tok/s | α: {sweep_res['alpha']*100:.1f}% | Match: {is_identical}")

    avg_base_tps = float(np.mean(base_tps_list))
    avg_sweep_tps = float(np.mean(sweep_tps_list))
    avg_alpha = float(np.mean(alpha_list)) if alpha_list else 0.0
    speedup = avg_sweep_tps / avg_base_tps if avg_base_tps > 0 else 1.0
    invariance_pct = (invariance_count / len(samples)) * 100.0

    print(f"  ==> {domain_name} SUMMARY: Base={avg_base_tps:.2f} tok/s | SWEEP={avg_sweep_tps:.2f} tok/s | Speedup={speedup:.2f}x | Acceptance={avg_alpha*100:.1f}% | Invariance={invariance_pct:.1f}%")
    return {
        'baseline_tps': avg_base_tps,
        'sweep_tps': avg_sweep_tps,
        'speedup': speedup,
        'acceptance_rate': avg_alpha,
        'invariance_pct': invariance_pct,
        'num_samples': len(samples)
    }

# ==============================================================================
# BENCHMARK PART 2: Context-Length Horizon Scaling (128 -> 4000 tokens)
# ==============================================================================
def run_context_scaling():
    print("\n" + "=" * 85)
    print("  PART 2: CONTEXT-LENGTH HORIZON SCALING BENCHMARK (N_ctx = 128 to 4000)")
    print("=" * 85)
    context_lengths = [128, 512, 1024, 2048, 3800]
    scaling_results = {}

    filler_text = "The quick brown fox jumps over the lazy dog. Academic research in deep learning requires rigorous verification and memory bandwidth profiling. "
    filler_tokens = tokenize_str(filler_text)

    for target_ctx in context_lengths:
        needed = max(target_ctx - 64, 16)
        prompt_toks = (filler_tokens * (needed // len(filler_tokens) + 1))[:needed]

        # Baseline
        _, base_t, base_tps = run_baseline_sequential(prompt_toks, n_predict=64, n_ctx=4096)
        # SWEEP
        sweep_res = run_sweep_wavefront(prompt_toks, n_predict=64, grammar_prior=grammar_prior, n_ctx=4096)
        sweep_tps = sweep_res['tokens_per_sec']
        speedup = sweep_tps / base_tps if base_tps > 0 else 1.0

        scaling_results[target_ctx] = {
            'prompt_length': len(prompt_toks),
            'baseline_tps': base_tps,
            'sweep_tps': sweep_tps,
            'speedup': speedup,
            'alpha': sweep_res['alpha']
        }
        print(f"  N_ctx = {target_ctx:>4} (prompt: {len(prompt_toks):>4} tok) | Baseline: {base_tps:.2f} tok/s | SWEEP: {sweep_tps:.2f} tok/s | Speedup: {speedup:.2f}x | α: {sweep_res['alpha']*100:.1f}%")

    return scaling_results

# ==============================================================================
# BENCHMARK PART 3: Real-Time Power Telemetry & Energy Efficiency (Joules/token)
# ==============================================================================
class PowerMonitor:
    def __init__(self, interval_sec: float = 0.05):
        self.interval = interval_sec
        self.running = False
        self.power_samples = []
        self.temp_samples = []
        self.thread = None

    def _poll(self):
        while self.running:
            try:
                res = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=power.draw,temperature.gpu', '--format=csv,noheader,nounits'],
                    stderr=subprocess.DEVNULL
                ).decode('utf-8').strip()
                pwr_str, temp_str = res.split(',')
                self.power_samples.append(float(pwr_str.strip()))
                self.temp_samples.append(float(temp_str.strip()))
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.power_samples = []
        self.temp_samples = []
        self.running = True
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def stop(self) -> Tuple[float, float]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        avg_power = float(np.mean(self.power_samples)) if self.power_samples else 0.0
        avg_temp = float(np.mean(self.temp_samples)) if self.temp_samples else 0.0
        return avg_power, avg_temp

def run_power_telemetry():
    print("\n" + "=" * 85)
    print("  PART 3: REAL-TIME POWER & ENERGY EFFICIENCY TELEMETRY (JOULES PER TOKEN)")
    print("=" * 85)
    prompt_text = format_chat_prompt("Write a detailed academic analysis comparing Von Neumann bottleneck and memory wall in modern deep learning accelerators.")
    prompt_tokens = tokenize_str(prompt_text)

    monitor = PowerMonitor()

    # 1. Baseline Run
    print("  Measuring Baseline GPU continuous generation (128 tokens)...")
    monitor.start()
    base_toks, base_time, base_tps = run_baseline_sequential(prompt_tokens, n_predict=128)
    base_power, base_temp = monitor.stop()
    base_joules_per_tok = base_power / base_tps if base_tps > 0 else 0.0
    print(f"  -> Baseline: {base_tps:.2f} tok/s | Avg Power: {base_power:.1f} W | Temp: {base_temp:.1f} C | Energy: {base_joules_per_tok:.3f} J/token")

    time.sleep(2.0) # Cool down

    # 2. SWEEP Run
    print("  Measuring SWEEP GPU continuous generation (128 tokens)...")
    monitor.start()
    sweep_res = run_sweep_wavefront(prompt_tokens, n_predict=128, grammar_prior=grammar_prior)
    sweep_power, sweep_temp = monitor.stop()
    sweep_tps = sweep_res['tokens_per_sec']
    sweep_joules_per_tok = sweep_power / sweep_tps if sweep_tps > 0 else 0.0
    energy_reduction_pct = (1.0 - (sweep_joules_per_tok / base_joules_per_tok)) * 100.0 if base_joules_per_tok > 0 else 0.0
    print(f"  -> SWEEP:    {sweep_tps:.2f} tok/s | Avg Power: {sweep_power:.1f} W | Temp: {sweep_temp:.1f} C | Energy: {sweep_joules_per_tok:.3f} J/token")
    print(f"  ==> ENERGY EFFICIENCY GAIN: {energy_reduction_pct:.1f}% reduction in Joules/token!")

    return {
        'baseline': {
            'tps': base_tps,
            'avg_power_watts': base_power,
            'avg_temp_c': base_temp,
            'joules_per_token': base_joules_per_tok
        },
        'sweep': {
            'tps': sweep_tps,
            'avg_power_watts': sweep_power,
            'avg_temp_c': sweep_temp,
            'joules_per_token': sweep_joules_per_tok
        },
        'energy_reduction_pct': energy_reduction_pct
    }

# ==============================================================================
# MAIN EXECUTION ORCHESTRATOR
# ==============================================================================
print("\n[STARTING SUITE EXECUTION]")
academic_results = {}
academic_results['GSM8K'] = run_academic_domain("GSM8K", GSM8K_SAMPLES, "question", n_predict=48)
academic_results['HumanEval'] = run_academic_domain("HumanEval", HUMANEVAL_SAMPLES, "prompt", n_predict=48)
academic_results['BoolQ'] = run_academic_domain("BoolQ", BOOLQ_SAMPLES, "question", n_predict=32)

context_results = run_context_scaling()
power_results = run_power_telemetry()

# Master Results Compilation
master_results = {
    'hardware': {
        'gpu': 'NVIDIA GeForce RTX 5070 Ti',
        'architecture': 'Blackwell SM 12.0',
        'vram_capacity_gb': 16,
        'vram_type': 'GDDR7 (735 GB/s)',
        'cpu': 'Intel Core i5-14600KF (14 Cores / 20 Threads)'
    },
    'model': {
        'name': 'Meta-Llama-3-8B-Instruct',
        'quantization': 'Q4_K_M',
        'offload': '100% VRAM (32/32 layers on CUDA0)'
    },
    'academic_domains': academic_results,
    'context_scaling': context_results,
    'power_telemetry': power_results
}

out_file = '/workspace/sweep/benchmarks/rtx5070ti_definitive_benchmark_results.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(master_results, f, indent=2)

print("\n" + "=" * 85)
print(f"  DEFINITIVE BENCHMARK COMPLETE! RESULTS PERSISTED TO: {out_file}")
print("=" * 85)

libllama.llama_model_free(model_ptr)
libllama.llama_backend_free()
