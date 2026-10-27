import sys
import os
import time
import psutil
import torch
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = r"C:\Users\Umberto\synaptic-fold-v2"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_full28_model import SynapticFull28Model

snapshot_dir = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

print("=" * 80, flush=True)
print("SYNAPTICFOLD v2: FULL 28-LAYER NATIVE SYNAPTIC INFERENCE ON QWEN2.5-7B", flush=True)
print("Hardware Target: Physical RAM <= 5.18 GB | Pure AVX-2 INT4-MM | All 28 Layers", flush=True)
print("=" * 80, flush=True)

def on_progress(key, loaded, total):
    print(f"[{loaded}/{total}] Quantized & loaded: {key} (RSS: {psutil.Process().memory_info().rss / 1024**2:.1f} MB)", flush=True)

t0 = time.perf_counter()
model = SynapticFull28Model.from_pretrained(snapshot_dir, progress_callback=on_progress)
load_time = time.perf_counter() - t0
print(f"\nAll 28 layers quantized and loaded in {load_time:.2f} seconds!", flush=True)

mem = model.get_memory_breakdown()
print(f"Model Parameters in RAM: {mem['model_ram_mb']:.1f} MB ({mem['model_ram_gb']:.2f} GB)", flush=True)
print(f"Current Process RSS:     {mem['rss_mb']:.1f} MB ({mem['rss_gb']:.2f} GB)", flush=True)

# Sanity Check on Simple Greeting First
print("\n--- SANITY CHECK: GREETING ---", flush=True)
sanity_prompt = "<|im_start|>user\nCiao! Come stai?<|im_end|>\n<|im_start|>assistant\n"
sanity_ids = tokenizer(sanity_prompt, return_tensors="pt")["input_ids"]

def on_token_sanity(tok_id: int):
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
    print(tok_str, end="", flush=True)

with torch.no_grad():
    _ = model.generate(
        sanity_ids,
        max_new_tokens=25,
        temperature=0.0,
        token_callback=on_token_sanity,
    )
print("\n--- END OF SANITY CHECK ---\n", flush=True)

# Now Solve the Farmer-Wolf Riddle with the Twist
riddle = (
    "Un contadino deve attraversare un fiume con un cavolo, una pecora e un lupo. "
    "La sua barca può portare solo lui e un oggetto alla volta. "
    "Se lasciati soli, la pecora mangia il cavolo e il lupo mangia la pecora. "
    "Tuttavia, in questa versione il contadino ha una corda con cui ha legato saldamente il lupo a un albero sulla riva iniziale, "
    "rendendolo inoffensivo finché resta lì. "
    "Descrivi la sequenza di viaggi più rapida per portare tutto dall'altra parte."
)

messages = [{"role": "user", "content": riddle}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

print(f"Prompt length: {input_ids.shape[1]} tokens", flush=True)
print("Streaming response on full 28-layer Synaptic model...", flush=True)
print("-" * 50, flush=True)

generated_tokens = []
def on_token(tok_id: int):
    generated_tokens.append(tok_id)
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
    print(tok_str, end="", flush=True)

t0 = time.perf_counter()
with torch.no_grad():
    out_ids = model.generate(
        input_ids,
        max_new_tokens=150,
        temperature=0.2,
        top_k=40,
        token_callback=on_token,
    )
gen_time = time.perf_counter() - t0

new_tokens = len(generated_tokens)
tok_s = new_tokens / max(gen_time, 1e-4)
print(f"\n\nGenerated {new_tokens} tokens in {gen_time:.2f} s ({tok_s:.2f} tok/s)", flush=True)

mem_final = model.get_memory_breakdown()
print(f"Final Physical Parameters RAM: {mem_final['model_ram_mb']:.1f} MB ({mem_final['model_ram_gb']:.2f} GB)", flush=True)
print(f"Final Process RSS:             {mem_final['rss_mb']:.1f} MB ({mem_final['rss_gb']:.2f} GB)", flush=True)

response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
out_file = os.path.join(ROOT_DIR, "benchmarks", "7b_riddle_response.txt")
with open(out_file, "w", encoding="utf-8") as f:
    f.write(response_text)

print(f"Result saved to {out_file}", flush=True)
