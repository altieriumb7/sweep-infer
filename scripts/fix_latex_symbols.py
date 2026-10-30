import os

tex_path = r"C:\Users\Umberto\synaptic-fold-v2\sweep_definitive_paper.tex"

with open(tex_path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix the broken text in Introduction:
old_intro = r"""\section{Introduction}

The decoding phase of autoregressive transformer inference in generative artificial intelligence is characterized by extreme memory-bound behavior across all modern computing platforms. While the prompt processing (prefill) phase benefits from high compute density via batched matrix-matrix multiplication (GEMM), token-by-token generation is strictly governed by matrix-vector multiplication (GEMV). For an $-parameter quantized model (e.g., 4.5~GB for 8B parameters at 4.5~BPW), generating each individual token necessitates reading the entire .5$~GB parameter tensor from memory into arithmetic registers.

On general-purpose CPUs with DDR4/DDR5 buses, sustained memory bandwidth caps single-sequence throughput to \text{--}8$~tokens/s, leaving arithmetic ALUs severely starved. Crucially, discrete graphics processing units (GPUs) suffer from the exact same architectural bottleneck during single-sequence autoregression: even on modern GDDR6/GDDR7 and HBM buses providing hundreds of gigabytes per second, single-batch GEMV saturates memory bus bandwidth while massive Tensor Core compute pipelines sit predominantly idle. Previous attempts to overcome this trade-off have relied on weight pruning, structured matrix sparsification, or lossy distillation, which inevitably compromise logical reasoning, introduce mathematical hallucinations, and alter probability distributions. Meanwhile, classical speculative decoding requires transferring multi-gigabyte auxiliary draft models across memory and PCIe buses, causing severe bus congestion and net throughput regressions on memory-constrained systems.

In order to overcome this hardware limitation without degrading model accuracy, I introduce \textbf{SWEEP}, a unified inference paradigm that fuses low-dimensional latent state projection with entropy-controlled batched microarchitectural speculation across both CPUs and GPUs. The foundational insight is that streaming .5$~GB of weights from DRAM/VRAM for a batch size =1$ costs virtually the same physical transfer time as streaming the identical tensor for a multi-token wavefront (=K$). When speculative tokens are accepted, the parameter sweep is amortized across multiple emitted tokens, multiplying instantaneous throughput. If a speculative candidate is rejected, an instantaneous (1)$ rollback efficiently prunes the speculative key-value slots with \textit{zero execution penalty}."""

new_intro = r"""\section{Introduction}

The decoding phase of autoregressive transformer inference in generative artificial intelligence is characterized by extreme memory-bound behavior across all modern computing platforms. While the prompt processing (prefill) phase benefits from high compute density via batched matrix-matrix multiplication (GEMM), token-by-token generation is strictly governed by matrix-vector multiplication (GEMV). For an $N$-parameter quantized model (e.g., $4.5$~GB for 8B parameters at 4.5~BPW), generating each individual token necessitates reading the entire $4.5$~GB parameter tensor from memory into arithmetic registers.

On general-purpose CPUs with DDR4/DDR5 buses, sustained memory bandwidth caps single-sequence throughput to $4\text{--}8$~tokens/s, leaving arithmetic ALUs severely starved. Crucially, discrete graphics processing units (GPUs) suffer from the exact same architectural bottleneck during single-sequence autoregression: even on modern GDDR6/GDDR7 and HBM buses providing hundreds of gigabytes per second, single-batch GEMV saturates memory bus bandwidth while massive Tensor Core compute pipelines sit predominantly idle. Previous attempts to overcome this trade-off have relied on weight pruning, structured matrix sparsification, or lossy distillation, which inevitably compromise logical reasoning, introduce mathematical hallucinations, and alter probability distributions. Meanwhile, classical speculative decoding requires transferring multi-gigabyte auxiliary draft models across memory and PCIe buses, causing severe bus congestion and net throughput regressions on memory-constrained systems.

In order to overcome this hardware limitation without degrading model accuracy, I introduce \textbf{SWEEP}, a unified inference paradigm that fuses low-dimensional latent state projection with entropy-controlled batched microarchitectural speculation across both CPUs and GPUs. The foundational insight is that streaming $4.5$~GB of weights from DRAM/VRAM for a batch size $B=1$ costs virtually the same physical transfer time as streaming the identical tensor for a multi-token wavefront ($B=K$). When speculative tokens are accepted, the parameter sweep is amortized across multiple emitted tokens, multiplying instantaneous throughput. If a speculative candidate is rejected, an instantaneous $O(1)$ rollback efficiently prunes the speculative key-value slots with \textit{zero execution penalty}."""

# Normalize newlines for matching
text = text.replace(old_intro.replace("\r\n", "\n"), new_intro.replace("\r\n", "\n"))
text = text.replace(old_intro, new_intro)

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Repaired LaTeX intro formulas with exact dollar signs.")
