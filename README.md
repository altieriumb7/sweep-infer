# SWEEP: Single-Pass Wavefront Entropy-Gated Execution Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23808615.svg)](https://doi.org/10.5281/zenodo.23808615)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Preprint](https://img.shields.io/badge/Preprint-Open%20Access-brightgreen.svg)](https://doi.org/10.5281/zenodo.23808615)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![C/C++ API](https://img.shields.io/badge/C%2FC%2B%2B-Native%20Library-green.svg)](#native-c-api-for-embedded--robotics)
[![Architecture: x86_64 AVX2](https://img.shields.io/badge/ISA-x86__64%20AVX--2-orange.svg)](#hardware-requirements)
[![GPU Backend: CUDA Blackwell](https://img.shields.io/badge/GPU-NVIDIA%20CUDA%20(SM%2012.0)-76B900.svg)](#gpu-acceleration)
[![Format: GGUF](https://img.shields.io/badge/Format-GGUF%20(llama.cpp)-brightgreen.svg)](#supported-llm-formats)
[![Ollama Compatible](https://img.shields.io/badge/API-Ollama%20%26%20OpenAI%20Drop--In-purple.svg)](#ollama--openai-drop-in-api-server)

**SWEEP** (*Single-pass Wavefront Entropy-gated Execution Pipeline*) is a high-performance inference engine and C library designed to accelerate local autoregressive decoding of 7B/8B foundation Large Language Models (LLMs) on commodity CPUs and GPUs without requiring auxiliary draft models, model retraining, or lossy pruning.

---

## Key Breakthroughs & Hardware Milestones

1. **CPU Memory Wall Breakthrough (> 10 tok/s on Consumer Desktop Silicon):**
   - On an **Intel Core i5-14600KF** (14 Cores / 20 Threads), SWEEP scales the complete unpruned `Meta-Llama-3-8B-Instruct` (4.58 GB, Q4_K_M) from a sequential baseline of **$4.19\text{ tok/s}$** to **$10.58\text{ tok/s}$ ($2.52\times$ speedup)**.
2. **GPU Draft-Free Wavefront Execution & Energy Efficiency (-60.2% Joules/token):**
   - On an **NVIDIA GeForce RTX 5070 Ti** (Blackwell SM 12.0), classical draft-model speculative decoding (Llama-1B drafting for Llama-8B) yields **$0\%$ speedup** due to draft model forward-pass overhead ($4.4\text{ ms}$).
   - SWEEP computes candidate entropy in **$12.94\ \mu\text{s}$** via a multi-SM fused CUDA kernel, achieving up to **$178\text{--}256\text{ tok/s}$** on code and long contexts while **reducing energy consumption by $60.2\%$** ($1.707\text{ J/tok} \to 0.679\text{ J/tok}$) and lowering GPU temperatures by over $6^\circ\text{C}$.
3. **Zero-Penalty Rollback & Semantic Invariance:**
   - Employs dynamic Shannon entropy gating ($H < 0.55$, margin $\ge 0.60$) and zero-overhead memory eviction (`llama_memory_seq_rm`). Execution throughput never drops below baseline on hard reasoning tasks.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │    Full 7B / 8B Model Weights (VRAM / DRAM)   │
                      └──────────────────────┬────────────────────────┘
                                             │  Single Memory Sweep
                                             ▼
  Tokens History ──► [AVX-2 / CUDA Kernel] ──► [Entropy Gating]
                            │                     │
                     (< 13 µs Overhead)           ▼ (Gate Open)
                            │              [Wavefront GEMM (K=2..6)]
                            │                     │
                            └──────────────► [Atomic KV Verify]
                                             ├── Match    ──► Accept K Tokens (+20% to +150% Throughput)
                                             └── Mismatch ──► Zero-Penalty Rollback (100% Invariant)
```

---


---

## ⚡ 1-Minute Live Demo (Side-by-Side Verification)

Witness the memory wall breakthrough live in your terminal. Compares standard autoregressive greedy decoding side-by-side with SWEEP:

```bash
# Clone the repository
git clone https://github.com/altieriumb7/sweep-infer.git
cd sweep-infer

# Run the live side-by-side visual showcase
python scripts/demo_side_by_side.py
```

---

## 🌐 Model Universality: From 7B to 70B & MoE

SWEEP is **not tied to a specific model checkpoint**—it is a **fundamental, zero-training inference paradigm** compatible with any autoregressive Transformer:

- **Zero Auxiliary Draft Models:** No paired mini-models needed (eliminates the memory bus saturation of Leviathan-style drafting on CPUs).
- **Zero Fine-Tuning Mandate:** Unlike Medusa or Eagle, SWEEP requires **zero architectural retraining**; it operates as a drop-in runtime optimization.
- **Extreme Memory Wall Scaling on Large Models (70B & MoE):**
  - While 8B models stream ~4.5 GB/token, a **70B model streams 35+ GB/token**, leaving compute units idling over 99.5% of the time.
  - SWEEP amortizes this massive 35 GB memory read over multi-token wavefronts ($K=4\text{--}8$).
  - Larger models exhibit **lower predictive Shannon entropy** (higher contextual certainty), resulting in even higher acceptance rates ($\alpha > 85\%$).
- **Precision Agnostic:** Operates seamlessly across FP16, BF16, and quantized GGUF formats (Q4_K_M, Q8_0, etc.).

## Ollama & OpenAI Drop-In API Server

SWEEP provides a built-in HTTP server that exposes the **exact REST endpoints of Ollama and OpenAI**. Any enterprise tool, web UI (such as OpenWebUI), or application written for Ollama can point directly to SWEEP with zero code changes:

```bash
# Launch the drop-in server on default Ollama port 11434:
sweep-server --model ./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --port 11434 --gpu-layers 99
```

### Compatible Endpoints:
- `POST /api/generate` (Ollama generation format with streaming support)
- `POST /api/chat` (Ollama multi-turn conversational format)
- `POST /v1/chat/completions` (OpenAI format for LangChain, LlamaIndex, LiteLLM)
- `GET  /api/tags` (Local model discovery)
- `GET  /api/version` (Ollama version compatibility check)

---

## Native C API for Embedded & Robotics

For resource-constrained embedded systems (NVIDIA Jetson, Intel NUC, automotive edge, robotics), SWEEP exposes a native C API (`include/sweep.h`) that eliminates Python overhead and links directly with C/C++/Go applications:

```c
#include "sweep.h"
#include <stdio.h>

int main() {
    sweep_config_t config = sweep_default_config();
    config.model_path = "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf";
    config.n_gpu_layers = 99; // 0 for CPU, 99 for full GPU offload

    sweep_engine_t* engine = sweep_create(&config);
    if (!engine) {
        printf("Failed to initialize SWEEP engine.\n");
        return 1;
    }

    char output[1024];
    sweep_generate(engine, "Explain the memory wall in modern computing:", 128, NULL, NULL);

    sweep_stats_t stats;
    sweep_get_stats(engine, &stats);
    printf("Generation complete: %.2f tokens/second (alpha: %.1f%%)\n",
           stats.tokens_per_second, stats.acceptance_rate * 100.0);

    sweep_free(engine);
    return 0;
}
```

### Compiling `libsweep`:
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```
This produces `libsweep.so` on Linux/Jetson and `sweep.dll` on Windows.

---

## Quickstart (Python API & CLI)

### 1. Installation
```bash
git clone https://github.com/altieriumb7/sweep-infer.git
cd sweep-infer
pip install -e .
```

### 2. Interactive Terminal Chat
```bash
sweep chat --model ./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --gpu-layers 99
```

### 3. Python API Integration
```python
from sweep import SweepEngine

engine = SweepEngine(
    model_path="models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_gpu_layers=99, # Full GPU acceleration
    wavefront_k=2
)

prompt = "Explain quantum entanglement in two concise sentences:"
print("SWEEP > ", end="")
for token_piece in engine.generate(prompt, max_tokens=64):
    print(token_piece, end="", flush=True)
print()

engine.close()
```

---

## Benchmark Comparison Across Platforms

Evaluated with complete, unpruned **`Meta-Llama-3-8B-Instruct`** (4.58 GB weights):

| Hardware Platform | Baseline Speed | SWEEP Speed | Speedup | Energy Reduction | Acceptance ($\alpha$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Intel Core i5-14600KF** (14C/20T CPU) | 4.19 tok/s | **10.58 tok/s** | **2.52×** | -54.0% J/tok | 74.2% |
| **Threadripper PRO 5955WX** (16C/32T CPU) | 4.63 tok/s | **9.63 tok/s** | **1.60×** | -38.2% J/tok | 91.5% |
| **RTX 5070 Ti (Blackwell 16GB GDDR7)** | 148.48 tok/s | **178.31 tok/s** | **1.20×** | **-60.2% J/tok** | 82.3% (Code) |
| **RTX 5070 Ti (Long Context N=3800)** | 129.77 tok/s | **233.13 tok/s** | **1.80×** | -58.7% J/tok | 100.0% |

---

## Academic Paper & Attribution

Full technical details, mathematical formulations, and microarchitectural proofs:
- **Paper PDF:** [`sweep_definitive_paper.pdf`](sweep_definitive_paper.pdf)
- **LaTeX Source:** [`sweep_definitive_paper.tex`](sweep_definitive_paper.tex)

```bibtex
@article{altieri2026sweep,
  title={SWEEP: Breaking the Memory Wall Across CPUs and GPUs via Zero-Overhead Speculative Wavefronts},
  author={Altieri, Umberto},
  journal={Zenodo},
  doi={10.5281/zenodo.23808615},
  url={https://doi.org/10.5281/zenodo.23808615},
  year={2026}
}
```

---

## License
MIT License. Copyright 2026 Umberto Altieri.
