"""
SynapticFold: Training Pipeline for Non-Linear Cortical Manifold Adapter.

Trains a gated SwiGLU bottleneck adapter between L12 (Canonical Cortical Column)
and L26 (Motor Output Gateway) to eliminate the manifold discontinuity and align
representations with Italian natural language without modifying any 7B weights.
"""

import os
import sys
import json
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_column_7b import SynapticCorticalColumn7B, CorticalColumn7BConfig
from synaptic_fold.cortical_adapter import CorticalManifoldAdapter

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
CORPUS_PATH = os.path.join(ROOT_DIR, "synaptic_fold", "data", "cortical_calibration_corpus.json")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "synaptic_fold", "weights", "cortical_manifold_adapter.pt")


def train_manifold_adapter(
    num_epochs: int = 5,
    lr: float = 8e-4,
    bottleneck_dim: int = 1024,
    device: str = "cpu",
):
    print("=" * 75, flush=True)
    print("🧠 SYNAPTIC CORTICAL MANIFOLD ADAPTER: TRAINING PIPELINE", flush=True)
    print("=" * 75, flush=True)

    # 1. Load Tokenizer
    print("\n[1/6] Caricamento Tokenizer Qwen2.5...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)

    # 2. Load Corpus
    print(f"[2/6] Caricamento Corpus di Calibrazione da: {CORPUS_PATH}...", flush=True)
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  Totale campioni nel corpus: {len(corpus)}", flush=True)

    # 3. Load 5-Layer Model
    print("\n[3/6] Inizializzazione SynapticCorticalColumn7B (5 Layer BFloat16)...", flush=True)
    cfg = CorticalColumn7BConfig(dtype=torch.bfloat16, device=device)
    model = SynapticCorticalColumn7B.from_pretrained(SNAPSHOT_7B, config=cfg)

    # 4. Attach Non-Linear Adapter
    print(f"\n[4/6] Inizializzazione CorticalManifoldAdapter (Bottleneck: {bottleneck_dim})...", flush=True)
    adapter = CorticalManifoldAdapter(hidden_size=cfg.hidden_size, bottleneck_dim=bottleneck_dim, dtype=torch.bfloat16)
    model.bridge = adapter

    # Freeze 7B, unfreeze Adapter
    for p in model.parameters():
        p.requires_grad = False
    for p in model.bridge.parameters():
        p.requires_grad = True

    trainable_params = sum(p.numel() for p in model.bridge.parameters() if p.requires_grad)
    print(f"  Parametri da addestrare (Adapter): {trainable_params / 1e6:.2f} M ({trainable_params * 2 / 1024**2:.1f} MB in BF16)", flush=True)

    rss = psutil.Process().memory_info().rss / (1024**2)
    print(f"  RAM Attuale Processo: {rss:.1f} MB ({rss / 1024:.2f} GB) [Tetto: 5.18 GB]", flush=True)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.bridge.parameters(), lr=lr, weight_decay=1e-2)

    # Prepare tokenized data
    samples = []
    for item in corpus:
        msgs = [
            {"role": "user", "content": item["prompt"]},
            {"role": "assistant", "content": item["completion"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False)
        enc = tokenizer(text, return_tensors="pt")["input_ids"]
        samples.append(enc)

    # 6. Training Loop
    print("\n[5/6] Inizio Addestramento e Allineamento del Manifold...", flush=True)
    print("-" * 75, flush=True)

    total_steps = num_epochs * len(samples)
    global_step = 0
    start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        t_epoch_start = time.perf_counter()

        for idx, tokens in enumerate(samples):
            global_step += 1
            t_step = time.perf_counter()

            # Forward pass
            logits = model(tokens, last_logit_only=False)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = tokens[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, cfg.vocab_size), shift_labels.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.bridge.parameters(), max_norm=1.0)
            optimizer.step()

            step_time = time.perf_counter() - t_step
            epoch_loss += loss.item()

            print(
                f"  Epoch [{epoch}/{num_epochs}] | Step [{global_step}/{total_steps}] | "
                f"Loss: {loss.item():.4f} | Grad: {grad_norm:.3f} | Tempo: {step_time:.2f}s | "
                f"RAM: {psutil.Process().memory_info().rss / (1024**2):.0f} MB",
                flush=True,
            )

        avg_loss = epoch_loss / len(samples)
        epoch_time = time.perf_counter() - t_epoch_start
        print(f"==> Fine Epoch {epoch} | Loss Media: {avg_loss:.4f} | Tempo Epoch: {epoch_time:.1f}s\n", flush=True)

    total_time = time.perf_counter() - start_time
    print("-" * 75, flush=True)
    print(f"🎉 Addestramento completato in {total_time:.1f}s!", flush=True)

    # 7. Save Weights
    print(f"\n[6/6] Salvataggio pesi dell'adattatore in: {WEIGHTS_PATH}...", flush=True)
    model.bridge.save_weights(WEIGHTS_PATH)
    file_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024**2)
    print(f"  Pesi salvati con successo! Dimensione file: {file_size_mb:.2f} MB", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("✅ MANIFOLD ADAPTER PRONTO PER L'INFERENZA NATIVA!", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    train_manifold_adapter(num_epochs=4, lr=1e-3, bottleneck_dim=1024)
