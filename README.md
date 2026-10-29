# SWEEP: Single-Pass Wavefront Entropy-Gated Execution Pipeline

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: x86_64 AVX2](https://img.shields.io/badge/ISA-x86__64%20AVX--2-orange.svg)](#hardware-requirements)
[![LLM Format: GGUF](https://img.shields.io/badge/Format-GGUF%20(llama.cpp)-brightgreen.svg)](#supported-llm-formats)
[![Equivalence: 100.0% Exact](https://img.shields.io/badge/Output-100.0%25%20Bit--to--Bit%20Identical-purple.svg)](#academic-results)

**SWEEP** (*Single-pass Wavefront Entropy-gated Execution Pipeline*) is an algorithmic and microarchitectural inference framework designed to accelerate local autoregressive decoding of foundation 7B/8B Large Language Models on commodity CPUs without weight pruning, retraining, or loss of mathematical accuracy.

---

## Hardware Requirements & Cross-Platform Support

### Can SWEEP run on any machine?
* **x86-64 Multi-Core CPUs (Maximum Speedup):**
  * SWEEP contains native assembly kernels optimized with **AVX-2**, **FMA3**, and hardware **CRC32** instructions.
  * **Supported CPUs:** AMD Ryzen / Threadripper / EPYC (Zen 1/2/3/4/5, 2017+) and Intel Core / Xeon (Haswell 4th Gen 2013 through Core Ultra).
  * **Latency:** All four assembly kernels execute in **$\approx 148\ \mu\text{s}$**, consuming **$< 0.09\%$** of a standard DDR4/DDR5 DRAM memory sweep cycle.
* **Universal Cross-Platform Fallback (ARM64 / Apple Silicon / Any CPU):**
  * If compiled AVX-2 binaries are not present (e.g., on Apple Silicon M1/M2/M3/M4, ARM64 servers like AWS Graviton, or older x86 chips), SWEEP automatically falls back to an exact, fully vectorized **NumPy/C reference engine**.
  * Ensures 100% mathematical fidelity and functional execution across any platform supporting Python 3.9+.
* **Memory (RAM) Footprint:**
  * Requires sufficient system RAM to hold the quantized model weights and KV cache.
  * A 7B/8B model in 4-bit requires **$\approx 4.5\text{ GB}$ of DRAM**. Minimum recommended RAM: **8 GB -- 16 GB** (DDR4 or DDR5).

---

## Supported LLM Formats & Quantizations

### What models and formats does SWEEP require?
* **Format:** **GGUF** (the canonical binary format developed by `llama.cpp` and used natively by Ollama, LM Studio, and Jan).
* **Quantization Levels:** 
  * Compatible with **all standard GGUF quantizations**: `Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q8_0`, and `F16`.
  * **Zero Modification of Weights:** SWEEP operates exclusively on hidden representations, prediction entropies, and associative KV cache management. It does **not** prune weights, alter tensors, or mandate quantization-aware re-training.
* **Target Foundation Models Tested & Certified:**
  1. `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` (8.03B parameters, 128k context vocab)
  2. `Qwen2.5-7B-Instruct.Q4_K_M.gguf` (7.61B parameters, 152k context vocab)
  * Universally compatible with any causal autoregressive transformer supported by `llama.cpp` (Mistral 7B, Gemma 2 9B, Phi-3/4, DeepSeek-R1-Distill-7B/8B).
* **Zero Auxiliary Draft Model Needed:**
  * Unlike classical speculative decoding (Leviathan et al., 2023), SWEEP requires **no secondary draft model** (e.g., 0.5B or 1B), eliminating auxiliary DRAM bus traffic entirely.

---

## Key Benchmark Highlights (406 Academic Evaluations)

Evaluated across 7 canonical benchmark suites on an AMD Ryzen Threadripper PRO 5955WX (16 Cores, 32 Threads, 8-channel DDR4-3200):

| Suite | Benchmark Category | Acceptance Rate ($\alpha$) | Baseline (tok/s) | SWEEP (tok/s) | Speedup Factor | Bit-a-Bit Invariance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **HellaSwag** | Commonsense Continuation | **91.5%** | 6.00 | **9.63** | **+60.5% (1.60x)** | **100.0% Exact** |
| **HumanEval** | Python Code Synthesis | **88.5%** | 4.63 | **5.67** | **+22.5% (1.23x)** | **100.0% Exact** |
| **ARC-Challenge**| Scientific Reasoning | **74.0%** | 4.67 | **5.59** | **+19.7% (1.20x)** | **100.0% Exact** |
| **GSM8K** | Multi-Step Math | **78.3%** | 4.63 | **5.50** | **+18.8% (1.19x)** | **100.0% Exact** |
| **MMLU-STEM** | Collegiate Science & CS | **60.2%** | 4.65 | **5.44** | **+17.0% (1.17x)** | **100.0% Exact** |
| **MBPP** | Elementary Python Tasks | **70.4%** | 4.68 | **5.20** | **+11.1% (1.11x)** | **100.0% Exact** |
| **BoolQ** | Deductive Fact Comprehension | **34.0%** | 4.66 | **4.98** | **+6.9% (1.07x)** | **100.0% Exact** |

* **Zero Degradation:** Under low-speculative regimes (e.g. BoolQ), SWEEP's gated rollback ensures execution speed never falls below 0.992x ($\approx 1.0\times$ baseline).

---

## Architectural Comparison with Prior Work

| Dimension | Draft-Model Spec. (Leviathan '23) | Medusa / EAGLE | Prompt Lookup (PLD) | LayerSkip (Early Exit) | **SWEEP (Proposed)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Auxiliary Parameter Size** | 1.0 - 2.5 GB (Draft Model) | 50 - 200 MB (Heads) | None (0 MB) | None (0 MB) | **None ($< 1$ MB Metadata)** |
| **Retraining / Distillation** | None | **Required (Heavy)** | None | **Required** | **None (Zero Training)** |
| **Universal GGUF Compatibility**| High | Low | High | Low | **100% Plug-and-Play** |
| **CPU Memory Wall Impact** | **Severe Saturation** | Moderate | Low | Low | **Single Memory Sweep** |
| **Uncertainty-Aware Gating** | Rare (Static $K$) | Tree-Pruned | None (Brute Force) | Entropy-Gated | **AVX-2 Entropy + Margin** |
| **Misprediction Penalty** | High (Full Rerun) | Medium | High | Low | **Zero-Penalty Rollback** |
| **Output Numerical Invariance** | Exact | Approximate | Exact | Approximate | **100.0% Exact (406/406)** |
| **Kernel Overhead per Step** | $> 25\%$ DRAM Time | $> 15\%$ DRAM Time | $\approx 2\text{--}5\%$ | $\approx 3\%$ | **$< 0.09\%$ ($148\ \mu$s)** |

---

## Installation & Build

### 1. Clone & Setup Environment
```bash
git clone https://github.com/altieriumb7/synaptic-fold.git sweep
cd sweep
pip install -r requirements.txt
```

### 2. Compile AVX-2 SIMD Kernels (Linux / Windows)
To compile the high-performance AVX-2 assembly kernels:
```bash
# On Linux (GCC / Clang):
cd synaptic_fold/kernels
gcc -O3 -mavx2 -mfma -msse4.2 -shared -fPIC synaptic_tree_avx2.c -o synaptic_tree_avx2.so
gcc -O3 -mavx2 -mfma -msse4.2 -shared -fPIC synaptic_entropy_topk_avx2.c -o synaptic_entropy_topk_avx2.so
gcc -O3 -mavx2 -mfma -msse4.2 -shared -fPIC synaptic_ngram_router_avx2.c -o synaptic_ngram_router_avx2.so
gcc -O3 -mavx2 -mfma -msse4.2 -shared -fPIC synaptic_column_avx2.c -o synaptic_column_avx2.so
```

On Windows, the build script can be invoked using MSVC (`cl.exe /O2 /arch:AVX2`) or MinGW GCC. If the shared libraries are not found, SWEEP gracefully uses the vectorized NumPy fallback.

---

## Quickstart & Usage Guide

### 1. Model Preparation (Standard GGUF)
SWEEP operates directly on standard, unmodified GGUF quantization weights without requiring any auxiliary draft model.

Download the certified 7B/8B target models:
```bash
# Option A: Direct download via huggingface-cli
huggingface-cli download QuantFactory/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --local-dir ./models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ./models

# Option B: Run via standard Ollama local service
ollama pull llama3:8b
ollama pull qwen2.5:7b
```

### 2. High-Throughput In-Process Generation
To execute direct in-process inference utilizing compiled AVX-2 kernels and zero-overhead memory sweeps:
```bash
python benchmark_inprocess_wavefront.py --model ./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
```

### 3. Python API Integration
Integrate the SWEEP entropy-gated wavefront engine into any Python pipeline:
```python
from synaptic_fold.synaptic_wavefront_engine import ProprietaryWavefrontEngine

# 1. Initialize the SWEEP execution engine
engine = ProprietaryWavefrontEngine(server_url="http://127.0.0.1:8089")

# 2. Execute speculative generation with guaranteed bit-level invariance
prompt = "Explain why local LLM inference on CPUs is bounded by memory bandwidth rather than compute:"
result = engine.generate(
    prompt=prompt,
    max_tokens=150,
    temperature=0.0  # Greedy decoding guarantees 100% exact numerical identity
)

print("\n--- Generated Output ---")
print(result['text'])
print(f"\nThroughput: {result.get('speed_tok_s', 0.0):.2f} tokens/second")
```

### 4. Reproducing Academic Benchmarks (406 Evaluations)
To independently verify the results published in the paper across all 7 canonical suites (GSM8K, HumanEval, MBPP, ARC-Challenge, MMLU-STEM, HellaSwag, BoolQ):
```bash
# Run the complete master benchmark suite:
python benchmarks/benchmark_paper_suite.py --model ./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --all

# Run extended suite benchmarks (MBPP, ARC, MMLU-STEM, HellaSwag):
python benchmarks/benchmark_paper_suite_extended.py

# Generate publication-grade figures (Figures 1 to 4):
python benchmarks/generate_paper_charts.py
```

---

## Academic Paper & Citations

The full technical paper detailing the formal mathematical proofs, microarchitectural telemetry, and benchmark evaluation is available in:
* **PDF Deliverable:** [`sweep_definitive_paper.pdf`](sweep_definitive_paper.pdf)
* **LaTeX Source:** [`sweep_definitive_paper.tex`](sweep_definitive_paper.tex)

```bibtex
@article{altieri2026sweep,
  title={SWEEP: Single-Pass Wavefront Entropy-Gated Execution Pipeline for Accelerated Foundation LLM Inference on CPUs},
  author={Altieri, Umberto},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License
Apache License 2.0. Copyright 2026 Umberto Altieri.
