import os
import sys
import time
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_anytime_cortical_engine import SynapticAnytimeCorticalEngine

SYNAPSE_FILE = os.path.join(ROOT_DIR, "models", "qwen2.5_7b_cortical.synapse")
SNAPSHOT_DIR = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

print("=" * 80, flush=True)
print("BENCHMARK: SYNAPTIC ANYTIME CORTICAL ENGINE (CALIBRATED DEPTH ROUTING)", flush=True)
print("Technology: Native .synapse INT4 SIMD + Umberto's Anytime Depth Routing (IEEE 2026)", flush=True)
print("Hardware: AMD Ryzen 7 7730U | RAM Budget <= 5.18 GB", flush=True)
print("=" * 80, flush=True)

# 1. Load Model from native .synapse binary
t0 = time.perf_counter()
print(f"Loading SynapticAnytimeCorticalEngine from {SYNAPSE_FILE}...", flush=True)
model = SynapticAnytimeCorticalEngine.from_synapse(SYNAPSE_FILE)
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.2f} s via zero-copy native binary streaming!", flush=True)

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_DIR, local_files_only=True)

mem = model.get_memory_breakdown()
print(f"Memory: Model RAM {mem['model_ram_mb']:.1f} MB ({mem['model_ram_gb']:.2f} GB) | RSS {mem['rss_mb']:.1f} MB ({mem['rss_gb']:.2f} GB)", flush=True)
print(f"RAM Compliance: {'PASSED (<= 5.18 GB)' if mem['rss_gb'] <= 5.18 else 'WARNING'}\n", flush=True)

# 2. Test 1: Sanity Greeting
print("--- TEST 1: SANITY GREETING ---", flush=True)
prompt1 = "<|im_start|>user\nCiao! Come stai?<|im_end|>\n<|im_start|>assistant\n"
ids1 = tokenizer(prompt1, return_tensors="pt")["input_ids"]

t0_gen = time.perf_counter()
out1 = model.generate(ids1, max_new_tokens=25, temperature=0.0, top_k=1)
dt_gen1 = time.perf_counter() - t0_gen

response1 = tokenizer.decode(out1[0][ids1.shape[1]:], skip_special_tokens=True)
num_tok1 = out1.shape[1] - ids1.shape[1]
speed1 = num_tok1 / max(dt_gen1, 1e-4)

print(f"Output: {response1}")
print(f"Speed:  {num_tok1} tokens in {dt_gen1:.2f} s ({speed1:.2f} tok/s | {dt_gen1/max(num_tok1,1):.2f} s/tok)")
print(f"Depth Routing Telemetry:")
total_exits1 = sum(model.exit_counts.values()) or 1
for tier, count in sorted(model.exit_counts.items()):
    print(f"  Tier {tier:2d} (Depth {tier/28*100:4.1f}%): {count:2d} tokens ({count/total_exits1*100:4.1f}%)")
avg_depth1 = sum(tier * count for tier, count in model.exit_counts.items()) / total_exits1
print(f"  Average Layer Depth: {avg_depth1:.1f} / 28 layers (Saved {(1 - avg_depth1/28)*100:.1f}% layer FLOPs!)\n")

# 3. Test 2: The Logic Riddle
print("--- TEST 2: THE LOGIC RIDDLE ---", flush=True)
prompt2 = (
    "<|im_start|>user\n"
    "Un contadino deve trasportare un lupo, una capra e un cavolo dall'altra parte del fiume con una barca che può contenere solo lui e un solo oggetto. "
    "Come può fare senza che il lupo mangi la capra o la capra mangi il cavolo?<|im_end|>\n"
    "<|im_start|>assistant\n"
)
ids2 = tokenizer(prompt2, return_tensors="pt")["input_ids"]

t0_gen2 = time.perf_counter()
out2 = model.generate(ids2, max_new_tokens=50, temperature=0.0, top_k=1)
dt_gen2 = time.perf_counter() - t0_gen2

response2 = tokenizer.decode(out2[0][ids2.shape[1]:], skip_special_tokens=True)
num_tok2 = out2.shape[1] - ids2.shape[1]
speed2 = num_tok2 / max(dt_gen2, 1e-4)

print(f"Output:\n{response2}\n")
print(f"Speed:  {num_tok2} tokens in {dt_gen2:.2f} s ({speed2:.2f} tok/s | {dt_gen2/max(num_tok2,1):.2f} s/tok)")
print(f"Depth Routing Telemetry:")
total_exits2 = sum(model.exit_counts.values()) or 1
for tier, count in sorted(model.exit_counts.items()):
    print(f"  Tier {tier:2d} (Depth {tier/28*100:4.1f}%): {count:2d} tokens ({count/total_exits2*100:4.1f}%)")
avg_depth2 = sum(tier * count for tier, count in model.exit_counts.items()) / total_exits2
print(f"  Average Layer Depth: {avg_depth2:.1f} / 28 layers (Saved {(1 - avg_depth2/28)*100:.1f}% layer FLOPs!)")

print("=" * 80, flush=True)
print("BENCHMARK COMPLETED!", flush=True)
print("=" * 80, flush=True)
