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

from synaptic_fold.synaptic_native_model import SynapticNativeModel

snapshot_dir = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

print("=" * 80, flush=True)
print("LOADING REAL QWEN2.5-7B-INSTRUCT VIA SYNAPTIC NATIVE SELECTIVE LOADER", flush=True)
print("Target: Resident RAM <= 5.18 GB | 100% Native BFloat16 | Zero GGUF/INT4", flush=True)
print("=" * 80, flush=True)

try:
    t0 = time.perf_counter()
    model = SynapticNativeModel.from_pretrained(
        snapshot_dir,
        dtype=torch.bfloat16,
        device="cpu",
        load_real_weights=True,
    )
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.2f} s", flush=True)
except Exception as e:
    import traceback
    print("ERROR DURING LOADING:", flush=True)
    traceback.print_exc()
    sys.exit(1)

mem = model.get_memory_breakdown()
print(f"Physical Parameters in RAM: {mem['unique_physical_mb']:.1f} MB ({mem['unique_physical_mb']/1024:.2f} GB)", flush=True)
print(f"Process Resident Set Size:  {mem['rss_mb']:.1f} MB ({mem['rss_gb']:.2f} GB)", flush=True)
print(f"Compression Ratio:          {mem['compression_ratio']}x", flush=True)

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

print(f"\nPrompt length: {input_ids.shape[1]} tokens", flush=True)
print("Running fast autoregressive generation (KV-cached) on Qwen2.5-7B...", flush=True)
print("--- STREAMING RESPONSE ---", flush=True)

def on_token(tok_id: int):
    tok_str = tokenizer.decode([tok_id], skip_special_tokens=True)
    try:
        print(tok_str, end="", flush=True)
    except Exception:
        pass

t0 = time.perf_counter()
try:
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.0,
            use_cache=True,
            token_callback=on_token,
        )
    gen_time = time.perf_counter() - t0
    print("\n--- END OF STREAM ---", flush=True)

    new_tokens = out_ids.shape[1] - input_ids.shape[1]
    tok_s = new_tokens / max(gen_time, 1e-4)
    print(f"Generated {new_tokens} tokens in {gen_time:.2f} s ({tok_s:.2f} tok/s)", flush=True)

    rss_final = psutil.Process().memory_info().rss / (1024**2)
    print(f"Final Process RSS: {rss_final:.1f} MB ({rss_final/1024:.2f} GB)", flush=True)

    response_text = tokenizer.decode(out_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    out_file = os.path.join(ROOT_DIR, "benchmarks", "7b_riddle_response.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(response_text)

    print("\n" + "=" * 80, flush=True)
    print("QWEN2.5-7B SYNAPTIC NATIVE GENERATION RESULT:", flush=True)
    print("=" * 80, flush=True)
    try:
        print(response_text, flush=True)
    except Exception:
        print(response_text.encode("ascii", errors="replace").decode("ascii"), flush=True)
    print("=" * 80, flush=True)
    print(f"Full response saved to: {out_file}", flush=True)
except Exception as e:
    import traceback
    print("ERROR DURING GENERATION:", flush=True)
    traceback.print_exc()
