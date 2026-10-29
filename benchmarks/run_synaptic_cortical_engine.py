import os
import sys
import time
import gc
import psutil
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_engine import SynapticCorticalEngine, setup_optimal_cpu_threads

SNAPSHOT_DIR = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
SYNAPSE_FILE = os.path.join(ROOT_DIR, "models", "qwen2.5_7b_cortical.synapse")
os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)

print("=" * 80, flush=True)
print("BENCHMARK: SYNAPTIC CORTICAL ENGINE v2.5 ON QWEN2.5-7B", flush=True)
print("Target: Accelerate 7B generation on AMD Ryzen 7 7730U (16 Threads, Fused SIMD)", flush=True)
print("Constraint: Physical resident RAM strictly <= 5.18 GB (No Windows Error 1455)", flush=True)
print("=" * 80, flush=True)

setup_optimal_cpu_threads()
print(f"Configured PyTorch CPU threads: {torch.get_num_threads()}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_DIR, local_files_only=True)

# 1. Load Model
t0 = time.perf_counter()
if os.path.exists(SYNAPSE_FILE):
    print(f"Loading from pre-serialized native .synapse binary ({SYNAPSE_FILE})...", flush=True)
    model = SynapticCorticalEngine.from_synapse(SYNAPSE_FILE)
    load_time = time.perf_counter() - t0
    print(f"Loaded from .synapse in {load_time:.2f} s! (Blazingly fast streaming)", flush=True)
else:
    print(f"Building Fused Cortical Engine from safetensors snapshots...", flush=True)
    model = SynapticCorticalEngine.from_pretrained(SNAPSHOT_DIR)
    load_time = time.perf_counter() - t0
    print(f"Built & Fused in {load_time:.2f} s", flush=True)
    print(f"Serializing to native .synapse binary format for instant future loads...", flush=True)
    t_save = time.perf_counter()
    model.save_synapse(SYNAPSE_FILE)
    print(f"Saved .synapse in {time.perf_counter() - t_save:.2f} s", flush=True)

# 2. Check Memory Breakdown
mem = model.get_memory_breakdown()
print("\n" + "=" * 50, flush=True)
print(f"MEMORY PROFILE:")
print(f"  Model Parameters/Buffers RAM: {mem['model_ram_mb']:.1f} MB ({mem['model_ram_gb']:.2f} GB)")
print(f"  Process Resident Set Size:    {mem['rss_mb']:.1f} MB ({mem['rss_gb']:.2f} GB)")
print(f"  RAM Limit Compliance:        {'PASSED (<= 5.18 GB)' if mem['rss_gb'] <= 5.18 else 'WARNING (> 5.18 GB)'}")
print("=" * 50 + "\n", flush=True)

# 3. Test 1: Sanity Check
sanity_prompt = "<|im_start|>user\nCiao! Come stai?<|im_end|>\n<|im_start|>assistant\n"
sanity_ids = tokenizer(sanity_prompt, return_tensors="pt")["input_ids"]

print("--- TEST 1: SANITY CHECK (GREETING) ---", flush=True)
gen_tokens_sanity = []
def print_tok_sanity(tid):
    gen_tokens_sanity.append(tid)
    text = tokenizer.decode([tid], skip_special_tokens=True)
    print(text, end="", flush=True)

t0_gen = time.perf_counter()
_ = model.generate(
    sanity_ids,
    max_new_tokens=30,
    temperature=0.0,
    top_k=1,
    token_callback=print_tok_sanity,
)
t_gen_sanity = time.perf_counter() - t0_gen
tok_count_sanity = len(gen_tokens_sanity)
speed_sanity = tok_count_sanity / max(t_gen_sanity, 1e-4)
print(f"\n[Sanity Result: {tok_count_sanity} tokens in {t_gen_sanity:.2f} s ({speed_sanity:.2f} tok/s | {t_gen_sanity/max(tok_count_sanity,1):.2f} s/tok)]\n", flush=True)

# 4. Test 2: The Logic Riddle with Rope Variant
riddle_prompt = (
    "<|im_start|>user\n"
    "Un contadino deve attraversare un fiume con un lupo, una pecora e un cavolo. "
    "Ha una barca che può trasportare solo lui e un altro elemento alla volta. "
    "Tuttavia, il contadino ha una corda con cui può legare il lupo a un albero prima di iniziare qualsiasi operazione, "
    "impedendogli di mangiare la pecora. Come può trasportare tutto sull'altra riva nel minor numero di viaggi possibile? "
    "Descrivi la sequenza di viaggi.<|im_end|>\n"
    "<|im_start|>assistant\n"
)
riddle_ids = tokenizer(riddle_prompt, return_tensors="pt")["input_ids"]

print("--- TEST 2: LOGIC RIDDLE WITH ROPE VARIANT ---", flush=True)
gen_tokens_riddle = []
def print_tok_riddle(tid):
    gen_tokens_riddle.append(tid)
    text = tokenizer.decode([tid], skip_special_tokens=True)
    print(text, end="", flush=True)

t0_riddle = time.perf_counter()
output_ids = model.generate(
    riddle_ids,
    max_new_tokens=100,
    temperature=0.0,
    top_k=1,
    token_callback=print_tok_riddle,
)
t_gen_riddle = time.perf_counter() - t0_riddle
tok_count_riddle = len(gen_tokens_riddle)
speed_riddle = tok_count_riddle / max(t_gen_riddle, 1e-4)
print(f"\n[Riddle Result: {tok_count_riddle} tokens in {t_gen_riddle:.2f} s ({speed_riddle:.2f} tok/s | {t_gen_riddle/max(tok_count_riddle,1):.2f} s/tok)]\n", flush=True)

# Save result to benchmarks file
response_text = tokenizer.decode(output_ids[0][riddle_ids.shape[1]:], skip_special_tokens=True)
with open(os.path.join(ROOT_DIR, "benchmarks", "cortical_riddle_response.txt"), "w", encoding="utf-8") as f:
    f.write(response_text)

print("=" * 80, flush=True)
print("BENCHMARK COMPLETED SUCCESSFULLY!", flush=True)
print("=" * 80, flush=True)
