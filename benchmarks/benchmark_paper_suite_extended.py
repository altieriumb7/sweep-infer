"""
SynapticFold v2: Definitive Multi-Aspect Extended Academic Benchmark Suite
Datasets:
  1. MBPP (Mostly Basic Python Problems - 25 problems)
  2. ARC-Challenge (AI2 Reasoning Challenge - 25 questions)
  3. MMLU-STEM (CS, Math, Physics, Chemistry - 25 questions)
  4. HellaSwag (Commonsense Situational Continuation - 25 scenarios)
Target Platform: AMD Ryzen Threadripper PRO 5955WX (16 Cores, 32 Threads Zen 3)
"""

import ctypes
import os
import sys
import time
import json
import numpy as np
from typing import List, Tuple, Dict, Any

# Ensure stdout handles UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LIB_DIR = "/workspace/llama.cpp/build-cpu/bin"
libggml = ctypes.CDLL(os.path.join(LIB_DIR, "libggml.so"))
libllama = ctypes.CDLL(os.path.join(LIB_DIR, "libllama.so"))
libllama.llama_backend_init()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from synaptic_fold.kernels.native_router import (
    evaluate_speculative_tree_avx2,
    compute_column_potentials_avx2,
    compute_topk_entropy_avx2,
    init_ngram_router_avx2,
    seed_ngram_router_avx2,
    insert_ngram_router_avx2,
    query_ngram_router_avx2
)
from synaptic_fold.tree_speculator import SyntacticGrammarPrior
from benchmarks.benchmark_datasets_extended import (
    MBPP_SAMPLES,
    ARC_CHALLENGE_SAMPLES,
    MMLU_STEM_SAMPLES,
    HELLASWAG_SAMPLES
)

# Ctypes Prototypes
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

def tokenize_str(vocab_ptr, text: str) -> List[int]:
    b_text = text.encode('utf-8')
    tok_buf = (ctypes.c_int32 * (len(b_text) + 256))()
    n = libllama.llama_tokenize(vocab_ptr, b_text, len(b_text), tok_buf, len(tok_buf), False, True)
    return [tok_buf[i] for i in range(n)]

def token_to_str(vocab_ptr, token: int) -> str:
    buf = ctypes.create_string_buffer(128)
    rlen = libllama.llama_token_to_piece(vocab_ptr, token, buf, 128, 0, False)
    if rlen > 0:
        return buf.raw[:rlen].decode('utf-8', errors='replace')
    return ""

def format_chat_prompt(model_name: str, user_text: str) -> str:
    if "Llama" in model_name:
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        return f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"

def get_eos_tokens(model_name: str) -> set:
    if "Llama" in model_name:
        return {128001, 128008, 128009}
    else:
        return {151643, 151644, 151645}

class DominantBigramSpeculator:
    def __init__(self, prompt_tokens: List[int]):
        self.transitions = {}
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


def run_sequential_greedy(
    model_ptr, cparams, prompt_tokens: List[int], n_predict: int,
    vocab_size: int, eos_tokens: set
) -> Tuple[List[int], float, float]:
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    batch = libllama.llama_batch_init(16, 0, 1)

    batch_prefill = libllama.llama_batch_init(len(prompt_tokens) + 16, 0, 1)
    for i, tok in enumerate(prompt_tokens):
        batch_prefill.token[i] = tok
        batch_prefill.pos[i] = i
        batch_prefill.n_seq_id[i] = 1
        batch_prefill.seq_id[i][0] = 0
        batch_prefill.logits[i] = 1 if i == len(prompt_tokens) - 1 else 0
    batch_prefill.n_tokens = len(prompt_tokens)
    ret = libllama.llama_decode(ctx_ptr, batch_prefill)
    libllama.llama_batch_free(batch_prefill)
    assert ret == 0, f"Prefill failed: ret={ret}"

    l_ptr = libllama.llama_get_logits(ctx_ptr)
    idx, _, _, _ = compute_topk_entropy_avx2(l_ptr, vocab_size, 1)
    next_token = int(idx[0])

    generated_tokens = []
    curr_pos = len(prompt_tokens)

    t0 = time.perf_counter()
    while len(generated_tokens) < n_predict:
        if next_token in eos_tokens:
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

        generated_tokens.append(next_token)
        curr_pos += 1

        l_ptr = libllama.llama_get_logits(ctx_ptr)
        idx, _, _, _ = compute_topk_entropy_avx2(l_ptr, vocab_size, 1)
        next_token = int(idx[0])

    total_time = time.perf_counter() - t0
    tps = len(generated_tokens) / total_time if total_time > 0 else 0.0

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)
    return generated_tokens, total_time, tps


def run_synaptic_wavefront_speculative(
    model_ptr, cparams, prompt_tokens: List[int], n_predict: int,
    grammar_prior: SyntacticGrammarPrior, vocab_ptr,
    vocab_size: int, eos_tokens: set,
    entropy_gate_thresh: float = 0.55, margin_thresh: float = 0.60
) -> Dict[str, Any]:
    ctx_ptr = libllama.llama_init_from_model(model_ptr, cparams)
    mem_handle = libllama.llama_get_memory(ctx_ptr)
    batch = libllama.llama_batch_init(16, 0, 1)

    init_ngram_router_avx2()
    seed_ngram_router_avx2(prompt_tokens)
    bigram_speculator = DominantBigramSpeculator(prompt_tokens)

    batch_prefill = libllama.llama_batch_init(len(prompt_tokens) + 16, 0, 1)
    for i, tok in enumerate(prompt_tokens):
        batch_prefill.token[i] = tok
        batch_prefill.pos[i] = i
        batch_prefill.n_seq_id[i] = 1
        batch_prefill.seq_id[i][0] = 0
        batch_prefill.logits[i] = 1 if i == len(prompt_tokens) - 1 else 0
    batch_prefill.n_tokens = len(prompt_tokens)
    ret = libllama.llama_decode(ctx_ptr, batch_prefill)
    libllama.llama_batch_free(batch_prefill)
    assert ret == 0, f"Prefill failed: ret={ret}"

    l_ptr = libllama.llama_get_logits(ctx_ptr)
    idx, p, H_local, margin = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
    next_token = int(idx[0])
    history = list(prompt_tokens)

    curr_pos = len(prompt_tokens)
    generated_tokens = []
    topk_latencies_us = []
    tree_latencies_us = []
    col_latencies_us = []
    router_latencies_ns = []
    dram_sweep_times = []

    total_draft_proposed = 0
    total_draft_accepted = 0
    speculative_sweeps = 0
    decodes_count = 0

    D = 4096
    N_TREE = 5
    branch_weights = (np.ones((N_TREE, D), dtype=np.float32) / np.float32(np.sqrt(D))).astype(np.float32)
    branch_biases = np.zeros(N_TREE, dtype=np.float32)
    parent_indices = np.array([-1, 0, 0, 1, 2], dtype=np.int32)
    node_latents = np.ones((N_TREE, D), dtype=np.float32)

    t0_gen = time.perf_counter()

    while len(generated_tokens) < n_predict:
        if next_token in eos_tokens:
            break

        node_latents[0, :] = float(p[0])
        tk_tree = time.perf_counter()
        scores, sel_nodes, mask, n_sel = evaluate_speculative_tree_avx2(
            node_latents, branch_weights, branch_biases, parent_indices, prob_threshold=0.15
        )
        tree_latencies_us.append((time.perf_counter() - tk_tree) * 1e6)

        tk_col = time.perf_counter()
        col_pot = compute_column_potentials_avx2(node_latents[0], branch_weights[:4], branch_biases[:4], v_threshold=0.0)
        col_latencies_us.append((time.perf_counter() - tk_col) * 1e6)

        gate_open = (H_local < entropy_gate_thresh and margin >= margin_thresh) and (len(generated_tokens) + 2 <= n_predict)

        draft_tok = -1
        if gate_open and len(history) >= 1:
            tk_r = time.perf_counter()
            prev_tok = history[-1]
            draft_tok = query_ngram_router_avx2(prev_tok, next_token)
            if draft_tok == -1:
                draft_tok = bigram_speculator.query(next_token, min_dominance=0.55, min_count=2)
            router_latencies_ns.append((time.perf_counter() - tk_r) * 1e9)

            if draft_tok in eos_tokens:
                draft_tok = -1

            if draft_tok != -1:
                next_str = token_to_str(vocab_ptr, next_token)
                draft_str = token_to_str(vocab_ptr, draft_tok)
                if not grammar_prior.is_valid_transition(next_str, draft_str):
                    draft_tok = -1

        if draft_tok != -1:
            speculative_sweeps += 1
            total_draft_proposed += 1

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
            decodes_count += 1

            td0 = time.perf_counter()
            ret = libllama.llama_decode(ctx_ptr, batch)
            dram_sweep_times.append(time.perf_counter() - td0)
            if ret != 0:
                break

            tk0 = time.perf_counter()
            l0_ptr = libllama.llama_get_logits_ith(ctx_ptr, 0)
            idx0, p0, H0, m0 = compute_topk_entropy_avx2(l0_ptr, vocab_size, 5)
            topk_latencies_us.append((time.perf_counter() - tk0) * 1e6)
            pred_tok = int(idx0[0])

            generated_tokens.append(next_token)
            insert_ngram_router_avx2(history[-1], next_token, pred_tok)
            bigram_speculator.observe(next_token, pred_tok)
            history.append(next_token)

            if pred_tok == draft_tok:
                total_draft_accepted += 1
                generated_tokens.append(draft_tok)

                tk1 = time.perf_counter()
                l1_ptr = libllama.llama_get_logits_ith(ctx_ptr, 1)
                idx1, p1, H1, m1 = compute_topk_entropy_avx2(l1_ptr, vocab_size, 5)
                topk_latencies_us.append((time.perf_counter() - tk1) * 1e6)
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
                libllama.llama_memory_seq_rm(mem_handle, 0, curr_pos + 1, -1)
                curr_pos += 1
                next_token = pred_tok
                p = p0
                H_local = H0
                margin = m0
        else:
            batch.token[0] = next_token
            batch.pos[0] = curr_pos
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = 1
            batch.n_tokens = 1

            decodes_count += 1
            td0 = time.perf_counter()
            ret = libllama.llama_decode(ctx_ptr, batch)
            dram_sweep_times.append(time.perf_counter() - td0)
            if ret != 0:
                break

            generated_tokens.append(next_token)
            curr_pos += 1

            tk0 = time.perf_counter()
            l_ptr = libllama.llama_get_logits(ctx_ptr)
            idx, p, H_local, margin = compute_topk_entropy_avx2(l_ptr, vocab_size, 5)
            topk_latencies_us.append((time.perf_counter() - tk0) * 1e6)
            pred_tok = int(idx[0])

            insert_ngram_router_avx2(history[-1], next_token, pred_tok)
            bigram_speculator.observe(next_token, pred_tok)
            history.append(next_token)

            next_token = pred_tok

    total_time = time.perf_counter() - t0_gen
    tps = len(generated_tokens) / total_time if total_time > 0 else 0.0
    alpha = total_draft_accepted / max(1, total_draft_proposed)

    libllama.llama_batch_free(batch)
    libllama.llama_free(ctx_ptr)

    return {
        "generated_tokens": generated_tokens,
        "total_time": total_time,
        "tok_per_sec": tps,
        "decodes_count": decodes_count,
        "draft_proposed": total_draft_proposed,
        "draft_accepted": total_draft_accepted,
        "acceptance_rate": alpha,
        "speculative_sweeps": speculative_sweeps,
        "avg_topk_us": float(np.mean(topk_latencies_us)) if topk_latencies_us else 0.0,
        "avg_tree_us": float(np.mean(tree_latencies_us)) if tree_latencies_us else 0.0,
        "avg_col_us": float(np.mean(col_latencies_us)) if col_latencies_us else 0.0,
        "avg_router_ns": float(np.mean(router_latencies_ns)) if router_latencies_ns else 0.0,
        "avg_dram_sweep_ms": float(np.mean(dram_sweep_times) * 1e3) if dram_sweep_times else 0.0
    }


def evaluate_extended_suite_for_model(model_name: str, model_path: str, n_threads: int = 8) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"STARTING EXTENDED BENCHMARK: {model_name} (Threads: {n_threads})")
    print(f"Weights Path: {model_path}")
    print("=" * 90)

    mparams = libllama.llama_model_default_params()
    model_ptr = libllama.llama_model_load_from_file(model_path.encode('utf-8'), mparams)
    assert model_ptr is not None, f"Failed to load model {model_path}"

    vocab_ptr = libllama.llama_model_get_vocab(model_ptr)
    vocab_size = libllama.llama_vocab_n_tokens(vocab_ptr)
    eos_tokens = get_eos_tokens(model_name)
    print(f"Loaded Model Vocab Size: {vocab_size} | EOS Tokens: {eos_tokens}")

    cparams = libllama.llama_context_default_params()
    cparams.ints[0] = 4096      # n_ctx
    cparams.ints[7] = n_threads # n_threads
    cparams.ints[8] = n_threads # n_threads_batch

    grammar_prior = SyntacticGrammarPrior()

    results = {
        "model_name": model_name,
        "threads": n_threads,
        "vocab_size": vocab_size,
        "benchmark_mbpp": {},
        "benchmark_arc_challenge": {},
        "benchmark_mmlu_stem": {},
        "benchmark_hellaswag": {}
    }

    # 1. MBPP (Mostly Basic Python Problems - 25 problems)
    print(f"\n[Extended 1/4] MBPP Python Coding Benchmark ({len(MBPP_SAMPLES)} problems)")
    mbpp_matches = 0
    mbpp_alphas = []
    mbpp_speedups = []
    for idx, s in enumerate(MBPP_SAMPLES):
        prompt = format_chat_prompt(model_name, s['prompt'])
        tokens = tokenize_str(vocab_ptr, prompt)
        seq_toks, _, seq_tps = run_sequential_greedy(
            model_ptr, cparams, tokens, 32, vocab_size=vocab_size, eos_tokens=eos_tokens
        )
        syn_res = run_synaptic_wavefront_speculative(
            model_ptr, cparams, tokens, 32, grammar_prior, vocab_ptr, vocab_size=vocab_size, eos_tokens=eos_tokens
        )

        eq = (seq_toks == syn_res["generated_tokens"][:len(seq_toks)])
        if eq:
            mbpp_matches += 1
        mbpp_alphas.append(syn_res["acceptance_rate"])
        sp = syn_res["tok_per_sec"] / seq_tps if seq_tps > 0 else 1.0
        mbpp_speedups.append(sp)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(MBPP_SAMPLES):
            print(f"   [MBPP {idx+1:2d}/{len(MBPP_SAMPLES)}] Invariance: {mbpp_matches}/{idx+1} ({(mbpp_matches/(idx+1))*100:.1f}%) | Mean Alpha: {np.mean(mbpp_alphas)*100:.1f}% | Avg Speedup: {np.mean(mbpp_speedups):.2f}x")

    results["benchmark_mbpp"] = {
        "total": len(MBPP_SAMPLES),
        "bit_match_pct": (mbpp_matches / len(MBPP_SAMPLES)) * 100.0,
        "avg_alpha": float(np.mean(mbpp_alphas)),
        "avg_speedup": float(np.mean(mbpp_speedups))
    }

    # 2. ARC-Challenge (25 science reasoning questions)
    print(f"\n[Extended 2/4] ARC-Challenge Science Reasoning ({len(ARC_CHALLENGE_SAMPLES)} questions)")
    arc_matches = 0
    arc_alphas = []
    arc_speedups = []
    for idx, s in enumerate(ARC_CHALLENGE_SAMPLES):
        q_text = f"Question: {s['question']}\nChoices:\n{s['choices']}\nAnswer with the correct option letter:"
        prompt = format_chat_prompt(model_name, q_text)
        tokens = tokenize_str(vocab_ptr, prompt)
        seq_toks, _, seq_tps = run_sequential_greedy(
            model_ptr, cparams, tokens, 16, vocab_size=vocab_size, eos_tokens=eos_tokens
        )
        syn_res = run_synaptic_wavefront_speculative(
            model_ptr, cparams, tokens, 16, grammar_prior, vocab_ptr, vocab_size=vocab_size, eos_tokens=eos_tokens
        )

        eq = (seq_toks == syn_res["generated_tokens"][:len(seq_toks)])
        if eq:
            arc_matches += 1
        arc_alphas.append(syn_res["acceptance_rate"])
        sp = syn_res["tok_per_sec"] / seq_tps if seq_tps > 0 else 1.0
        arc_speedups.append(sp)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(ARC_CHALLENGE_SAMPLES):
            print(f"   [ARC {idx+1:2d}/{len(ARC_CHALLENGE_SAMPLES)}] Invariance: {arc_matches}/{idx+1} ({(arc_matches/(idx+1))*100:.1f}%) | Mean Alpha: {np.mean(arc_alphas)*100:.1f}% | Avg Speedup: {np.mean(arc_speedups):.2f}x")

    results["benchmark_arc_challenge"] = {
        "total": len(ARC_CHALLENGE_SAMPLES),
        "bit_match_pct": (arc_matches / len(ARC_CHALLENGE_SAMPLES)) * 100.0,
        "avg_alpha": float(np.mean(arc_alphas)),
        "avg_speedup": float(np.mean(arc_speedups))
    }

    # 3. MMLU-STEM (25 questions: CS, Math, Physics, Chemistry)
    print(f"\n[Extended 3/4] MMLU-STEM Multidisciplinary Reasoning ({len(MMLU_STEM_SAMPLES)} questions)")
    mmlu_matches = 0
    mmlu_alphas = []
    mmlu_speedups = []
    for idx, s in enumerate(MMLU_STEM_SAMPLES):
        q_text = f"Subject: {s['subject']}\nQuestion: {s['question']}\nOptions:\n{s['choices']}\nSelect the correct answer:"
        prompt = format_chat_prompt(model_name, q_text)
        tokens = tokenize_str(vocab_ptr, prompt)
        seq_toks, _, seq_tps = run_sequential_greedy(
            model_ptr, cparams, tokens, 16, vocab_size=vocab_size, eos_tokens=eos_tokens
        )
        syn_res = run_synaptic_wavefront_speculative(
            model_ptr, cparams, tokens, 16, grammar_prior, vocab_ptr, vocab_size=vocab_size, eos_tokens=eos_tokens
        )

        eq = (seq_toks == syn_res["generated_tokens"][:len(seq_toks)])
        if eq:
            mmlu_matches += 1
        mmlu_alphas.append(syn_res["acceptance_rate"])
        sp = syn_res["tok_per_sec"] / seq_tps if seq_tps > 0 else 1.0
        mmlu_speedups.append(sp)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(MMLU_STEM_SAMPLES):
            print(f"   [MMLU {idx+1:2d}/{len(MMLU_STEM_SAMPLES)}] Invariance: {mmlu_matches}/{idx+1} ({(mmlu_matches/(idx+1))*100:.1f}%) | Mean Alpha: {np.mean(mmlu_alphas)*100:.1f}% | Avg Speedup: {np.mean(mmlu_speedups):.2f}x")

    results["benchmark_mmlu_stem"] = {
        "total": len(MMLU_STEM_SAMPLES),
        "bit_match_pct": (mmlu_matches / len(MMLU_STEM_SAMPLES)) * 100.0,
        "avg_alpha": float(np.mean(mmlu_alphas)),
        "avg_speedup": float(np.mean(mmlu_speedups))
    }

    # 4. HellaSwag (25 commonsense situation continuation questions)
    print(f"\n[Extended 4/4] HellaSwag Commonsense Situational Continuation ({len(HELLASWAG_SAMPLES)} scenarios)")
    hs_matches = 0
    hs_alphas = []
    hs_speedups = []
    for idx, s in enumerate(HELLASWAG_SAMPLES):
        q_text = f"Activity: {s['activity']}\nContext: {s['context']}\nChoose the most plausible next action:\n{s['choices']}\nSelection:"
        prompt = format_chat_prompt(model_name, q_text)
        tokens = tokenize_str(vocab_ptr, prompt)
        seq_toks, _, seq_tps = run_sequential_greedy(
            model_ptr, cparams, tokens, 20, vocab_size=vocab_size, eos_tokens=eos_tokens
        )
        syn_res = run_synaptic_wavefront_speculative(
            model_ptr, cparams, tokens, 20, grammar_prior, vocab_ptr, vocab_size=vocab_size, eos_tokens=eos_tokens
        )

        eq = (seq_toks == syn_res["generated_tokens"][:len(seq_toks)])
        if eq:
            hs_matches += 1
        hs_alphas.append(syn_res["acceptance_rate"])
        sp = syn_res["tok_per_sec"] / seq_tps if seq_tps > 0 else 1.0
        hs_speedups.append(sp)

        if (idx + 1) % 5 == 0 or (idx + 1) == len(HELLASWAG_SAMPLES):
            print(f"   [HellaSwag {idx+1:2d}/{len(HELLASWAG_SAMPLES)}] Invariance: {hs_matches}/{idx+1} ({(hs_matches/(idx+1))*100:.1f}%) | Mean Alpha: {np.mean(hs_alphas)*100:.1f}% | Avg Speedup: {np.mean(hs_speedups):.2f}x")

    results["benchmark_hellaswag"] = {
        "total": len(HELLASWAG_SAMPLES),
        "bit_match_pct": (hs_matches / len(HELLASWAG_SAMPLES)) * 100.0,
        "avg_alpha": float(np.mean(hs_alphas)),
        "avg_speedup": float(np.mean(hs_speedups))
    }

    libllama.llama_model_free(ctypes.c_void_p(model_ptr))
    return results


def main():
    print("=" * 90)
    print("SYNAPTIC-FOLD V2: EXTENDED ACADEMIC BENCHMARK SUITE")
    print("Evaluates: MBPP (25), ARC-Challenge (25), MMLU-STEM (25), HellaSwag (25)")
    print("Platform: AMD Ryzen Threadripper PRO 5955WX (16C/32T Zen 3)")
    print("=" * 90)

    MODELS = {
        "Meta-Llama-3-8B-Instruct": "/workspace/models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-7B-Instruct": "/workspace/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    }

    suite_results = {
        "metadata": {
            "platform_cpu": "AMD Ryzen Threadripper PRO 5955WX 16-Core Processor",
            "threads_per_socket": 8,
            "suite_type": "Extended Canonical Academic Benchmarks"
        },
        "models": {}
    }

    for name, path in MODELS.items():
        if os.path.exists(path):
            suite_results["models"][name] = evaluate_extended_suite_for_model(name, path, n_threads=8)
        else:
            print(f"Skipping {name}: path not found at {path}")

    out_file = "/workspace/synaptic-fold-v2/benchmarks/paper_extended_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    print("\n" + "=" * 90)
    print(f"EXTENDED SCIENTIFIC SUITE SUCCESSFULLY EXECUTED! Data saved to: {out_file}")
    print("=" * 90)

if __name__ == "__main__":
    main()
