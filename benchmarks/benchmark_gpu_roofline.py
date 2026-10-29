import time
import torch

print('=' * 80)
print('ROOFLINE PROFILING SU NVIDIA GEFORCE RTX 5070 Ti (BLACKWELL 16GB GDDR7)')
print('Analisi GEMV (K=1) vs SWEEP Wavefront GEMM (K=2, 3, 4, 8) su Architettura 8B')
print('=' * 80)

device = torch.device('cuda:0')
gpu_name = torch.cuda.get_device_name(0)
print(f'Dispositivo Rilevato: {gpu_name}')

# 8B Model Architecture Dimensions (Llama-3-8B)
D_model = 4096
D_ffn = 14336
# A single transformer layer has:
# Q, K, V, O projections: 4 * 4096 * 4096
# Gate, Up, Down projections: 3 * 4096 * 14336
# Total weights per layer: 67,108,864 + 176,160,768 = 243,269,632 floats
# In FP16 / BF16: 486.5 MB per layer. For 32 layers = 15.5 GB (fits in 16 GB VRAM)
# Let's benchmark 4 representative transformer layers in FP16 (1.95 GB VRAM)
N_layers = 4
weight_bytes = N_layers * 243269632 * 2

print(f'\nAllocazione di {N_layers} layer completi FP16 ({weight_bytes / 1e9:.2f} GB VRAM)...')
torch.cuda.empty_cache()

layers = []
for _ in range(N_layers):
    w_qkv_o = torch.randn(4096, 4096 * 4, device=device, dtype=torch.float16)
    w_gate_up = torch.randn(4096, 14336 * 2, device=device, dtype=torch.float16)
    w_down = torch.randn(14336, 4096, device=device, dtype=torch.float16)
    layers.append((w_qkv_o, w_gate_up, w_down))

torch.cuda.synchronize()
print('Layer allocati con successo in VRAM GDDR7!')

# Warmup
for _ in range(20):
    x_warmup = torch.randn(1, 4096, device=device, dtype=torch.float16)
    for w1, w2, w3 in layers:
        h1 = torch.matmul(x_warmup, w1)
        h2 = torch.matmul(x_warmup, w2)
        h3 = torch.matmul(h2[:, :14336], w3)
torch.cuda.synchronize()

# Measure GEMV (K=1, Single-Token Sequential Decoding)
print('\n[1] Misurazione GEMV (K=1, Decodifica Sequenziale Baseline)...')
start_ev = torch.cuda.Event(enable_timing=True)
stop_ev = torch.cuda.Event(enable_timing=True)

N_ITERS = 100
start_ev.record()
for _ in range(N_ITERS):
    x = torch.randn(1, 4096, device=device, dtype=torch.float16)
    for w1, w2, w3 in layers:
        h1 = torch.matmul(x, w1)
        h2 = torch.matmul(x, w2)
        h3 = torch.matmul(h2[:, :14336], w3)
stop_ev.record()
torch.cuda.synchronize()

t_k1_ms = start_ev.elapsed_time(stop_ev) / N_ITERS
bw_k1 = (weight_bytes / 1e9) / (t_k1_ms / 1000.0)
print(f'  Latenza K=1 per {N_layers} layer: {t_k1_ms:.3f} ms | Bandwidth Effettiva VRAM: {bw_k1:.1f} GB/s')

# Measure SWEEP Wavefront GEMM (K=2, 3, 4, 8)
print('\n[2] Misurazione SWEEP Wavefront GEMM (K=2, 3, 4, 8)...')
results = {}
for k in [2, 3, 4, 8]:
    start_ev.record()
    for _ in range(N_ITERS):
        x = torch.randn(k, 4096, device=device, dtype=torch.float16)
        for w1, w2, w3 in layers:
            h1 = torch.matmul(x, w1)
            h2 = torch.matmul(x, w2)
            h3 = torch.matmul(h2[:, :14336], w3)
    stop_ev.record()
    torch.cuda.synchronize()

    t_k_ms = start_ev.elapsed_time(stop_ev) / N_ITERS
    ratio = t_k_ms / t_k1_ms
    lat_per_tok = t_k_ms / k
    raw_speedup = k / ratio

    # Speculative acceptance model
    alpha = 0.75
    eff_tokens = 1.0 + alpha * (k - 1)
    eff_speedup = eff_tokens / ratio

    results[k] = {
        'time_ms': t_k_ms,
        'ratio_to_k1': ratio,
        'ms_per_tok': lat_per_tok,
        'raw_speedup': raw_speedup,
        'eff_speedup': eff_speedup
    }
    print(f'  Wavefront K={k}: Sweep Time = {t_k_ms:.3f} ms ({ratio:.2f}x tempo K=1) | Latenza/tok = {lat_per_tok:.3f} ms | Raw Speedup = {raw_speedup:.2f}x | Eff. Speedup (alpha=0.75) = {eff_speedup:.2f}x')

print('\n' + '=' * 80)
print('DIMOSTRAZIONE EMPIRICA DEL SUPERAMENTO DEL MEMORY WALL SU GPU RTX 5070 Ti')
print('=' * 80)
print(f"{'Configurazione':<25} | {'Sweep Time (ms)':<15} | {'Delta vs K=1':<12} | {'Eff. Speedup'}")
print('-' * 80)
print(f"{'K=1 (Single Token GEMV)':<25} | {t_k1_ms:<15.3f} | 1.00x        | 1.00x")
for k, r in results.items():
    print(f"{'K=' + str(k) + ' (SWEEP Wavefront)':<25} | {r['time_ms']:<15.3f} | {r['ratio_to_k1']:<12.2f} | {r['eff_speedup']:.2f}x")
print('=' * 80)
