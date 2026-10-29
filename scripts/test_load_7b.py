import os
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

print("Test caricamento Qwen2.5-7B-Instruct con low_cpu_mem_usage...")
t0 = psutil.Process().memory_info().rss / (1024**2)
print(f"RAM iniziale: {t0:.1f} MB")

try:
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)
    # Check if we can load with bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT_7B,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
        local_files_only=True,
    )
    rss = psutil.Process().memory_info().rss / (1024**2)
    print(f"Modello 7B caricato con successo! RAM: {rss:.1f} MB ({rss/1024:.2f} GB)")
except Exception as e:
    print(f"Errore caricamento: {e}")
