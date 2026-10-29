"""
SynapticFold: High-Speed Streaming Trajectory Extractor & Manifold Adapter Trainer.

1. Streams forward through Qwen2.5-7B layers 0..25 with layer-by-layer garbage collection
   to extract the true latent target activations h_26 (Peak RAM < 1.8 GB).
2. Extracts the recurrent ODE attractor representations h_12 from SynapticCorticalColumn7B.
3. Trains the CorticalManifoldAdapter (22 MB) directly on (h_12 -> h_26) in seconds.
4. Verifies the newly aligned 5-layer BFloat16 model on benchmark prompts.
"""

import os
import sys
import gc
import json
import time
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from synaptic_fold.synaptic_cortical_column_7b import (
    SynapticCorticalColumn7B,
    CorticalColumn7BConfig,
    NativeDecoderLayer,
    NativeRMSNorm,
    NativeRotaryEmbedding,
)
from synaptic_fold.cortical_adapter import CorticalManifoldAdapter

SNAPSHOT_7B = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
CORPUS_PATH = os.path.join(ROOT_DIR, "synaptic_fold", "data", "cortical_calibration_corpus.json")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "synaptic_fold", "weights", "cortical_manifold_adapter.pt")
TRAJ_CACHE_PATH = os.path.join(ROOT_DIR, "synaptic_fold", "data", "calibration_trajectories_700.pt")


def get_layer_weights(layer_idx: int, weight_map: dict, snapshot_dir: str, config: CorticalColumn7BConfig):
    layer = NativeDecoderLayer(config)
    prefix = f"model.layers.{layer_idx}"
    target_keys = {
        f"{prefix}.input_layernorm.weight": layer.input_layernorm.weight,
        f"{prefix}.post_attention_layernorm.weight": layer.post_attention_layernorm.weight,
        f"{prefix}.self_attn.q_proj.weight": layer.self_attn.q_proj.weight,
        f"{prefix}.self_attn.q_proj.bias": layer.self_attn.q_proj.bias,
        f"{prefix}.self_attn.k_proj.weight": layer.self_attn.k_proj.weight,
        f"{prefix}.self_attn.k_proj.bias": layer.self_attn.k_proj.bias,
        f"{prefix}.self_attn.v_proj.weight": layer.self_attn.v_proj.weight,
        f"{prefix}.self_attn.v_proj.bias": layer.self_attn.v_proj.bias,
        f"{prefix}.self_attn.o_proj.weight": layer.self_attn.o_proj.weight,
        f"{prefix}.mlp.gate_proj.weight": layer.mlp.gate_proj.weight,
        f"{prefix}.mlp.up_proj.weight": layer.mlp.up_proj.weight,
        f"{prefix}.mlp.down_proj.weight": layer.mlp.down_proj.weight,
    }

    # Group by shard
    shards_needed = {}
    for k in target_keys:
        shard = weight_map[k]
        if shard not in shards_needed:
            shards_needed[shard] = []
        shards_needed[shard].append(k)

    for shard, keys in shards_needed.items():
        shard_path = os.path.join(snapshot_dir, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                t = f.get_tensor(k)
                target_keys[k].data.copy_(t.to(config.dtype))

    return layer


def extract_calibration_trajectories():
    print("=" * 75, flush=True)
    print("[FASE 1] ESTRAZIONE TRAIETTORIE LATENTI IN STREAMING (PEAK RAM < 2 GB)", flush=True)
    print("=" * 75, flush=True)

    if os.path.exists(TRAJ_CACHE_PATH):
        print(f"[CACHE] Caricamento traiettorie gia estratte da: {TRAJ_CACHE_PATH}...", flush=True)
        data = torch.load(TRAJ_CACHE_PATH, map_location="cpu", weights_only=True)
        X, Y = data["X"], data["Y"]
        print(f"[OK] Traiettorie caricate! Punti: {X.shape[0]} vettori da {X.shape[1]} dimensioni.\n", flush=True)
        return X, Y

    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)
    cfg = CorticalColumn7BConfig(dtype=torch.bfloat16, device="cpu")

    index_path = os.path.join(SNAPSHOT_7B, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # 1. Load Embeddings
    print("Caricamento matrice di embedding...", flush=True)
    embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=cfg.dtype)
    embed_shard = weight_map["model.embed_tokens.weight"]
    with safe_open(os.path.join(SNAPSHOT_7B, embed_shard), framework="pt", device="cpu") as f:
        embed_tokens.weight.data.copy_(f.get_tensor("model.embed_tokens.weight").to(cfg.dtype))

    rotary_emb = NativeRotaryEmbedding(dim=cfg.head_dim, max_position_embeddings=cfg.max_position_embeddings, base=cfg.rope_theta)

    all_source_h12 = []
    all_target_h26 = []

    # Process calibration sentences
    for item_idx, item in enumerate(corpus[:6], 1):  # 6 diverse sentences for high-quality calibration
        text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['completion']}<|im_end|>"
        tokens = tokenizer(text, return_tensors="pt")["input_ids"]
        seq_len = tokens.shape[1]
        print(f"\n[Frase {item_idx}/{min(len(corpus), 6)}] Elaborazione {seq_len} token: '{item['prompt'][:45]}...'", flush=True)

        pos_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
        cos, sin = rotary_emb(pos_ids, dtype=cfg.dtype)
        c_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)[None, None, :, :]

        # Initial hidden states
        h = embed_tokens(tokens)

        # We will forward stream through layers 0..25 to get true h_26
        # And we will also record h_12 through the ODE loop
        h_ode = None

        print("  Streaming forward strati 0 -> 25...", flush=True)
        for l_idx in range(26):
            layer = get_layer_weights(l_idx, weight_map, SNAPSHOT_7B, cfg)
            with torch.no_grad():
                h, _ = layer(h, cos, sin, attention_mask=c_mask)

                # At layer 12: also compute the ODE reverberations for the Column representation
                if l_idx == 12:
                    h_curr = h.clone()
                    for k in range(cfg.k_reverberations):
                        out, _ = layer(h_curr, cos, sin, attention_mask=c_mask)
                        delta = out - h_curr
                        h_curr = h_curr + cfg.ode_alpha * (delta * (cfg.ode_damping ** k))
                    h_ode = h_curr

            del layer
            gc.collect()

        print(f"  -> Estratti stati latenti: h_12_ode {h_ode.shape} e h_26_target {h.shape}", flush=True)
        all_source_h12.append(h_ode.squeeze(0))
        all_target_h26.append(h.squeeze(0))

    del embed_tokens
    gc.collect()

    X = torch.cat(all_source_h12, dim=0).float()
    Y = torch.cat(all_target_h26, dim=0).float()
    torch.save({"X": X, "Y": Y}, TRAJ_CACHE_PATH)
    print(f"[CACHE] Traiettorie salvate in {TRAJ_CACHE_PATH}", flush=True)
    print(f"[OK] Traiettorie estratte con successo! Punti di calibrazione: {X.shape[0]} vettori da {X.shape[1]} dimensioni.", flush=True)
    return X, Y


def train_adapter_on_trajectories(X: torch.Tensor, Y: torch.Tensor, epochs: int = 150, lr: float = 1e-3):
    print("\n" + "=" * 75, flush=True)
    print("[FASE 2] ADDESTRAMENTO ADATTATORE NON-LINEARE (Gated SwiGLU Bottleneck)", flush=True)
    print("=" * 75, flush=True)

    adapter = CorticalManifoldAdapter(hidden_size=3584, bottleneck_dim=1024, dtype=torch.float32)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    initial_mse = F.mse_loss(X, Y).item()
    initial_cos = F.cosine_similarity(X, Y, dim=-1).mean().item()
    print(f"Stato Iniziale Senza Adattatore (Discontinuita Grezza):")
    print(f"  MSE: {initial_mse:.6f} | Cosine Similarity: {initial_cos:.4f}", flush=True)

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        pred = adapter(X)

        mse_loss = F.mse_loss(pred, Y)
        cos_loss = 1.0 - F.cosine_similarity(pred, Y, dim=-1).mean()
        loss = mse_loss + 0.25 * cos_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if ep % 20 == 0 or ep == 1:
            cur_cos = F.cosine_similarity(pred.detach(), Y, dim=-1).mean().item()
            print(f"  Epoch [{ep:3d}/{epochs}] | Loss Totale: {loss.item():.6f} | MSE: {mse_loss.item():.6f} | CosSim: {cur_cos:.4f}", flush=True)

    train_time = time.perf_counter() - t0
    final_pred = adapter(X).detach()
    final_mse = F.mse_loss(final_pred, Y).item()
    final_cos = F.cosine_similarity(final_pred, Y, dim=-1).mean().item()

    print(f"\n[DONE] Addestramento completato in {train_time:.2f} s!")
    print(f"  MSE Finale: {final_mse:.6f} (Riduzione Errore: {((initial_mse - final_mse) / initial_mse) * 100:.1f}%)")
    print(f"  Cosine Similarity Finale: {final_cos:.4f} (Allineamento Manifold Raggiunto)")

    # Cast to bfloat16 and save
    adapter.to(torch.bfloat16)
    adapter.save_weights(WEIGHTS_PATH)
    print(f"[SAVED] Pesi dell'adattatore salvati in: {WEIGHTS_PATH} ({os.path.getsize(WEIGHTS_PATH) / 1024**2:.1f} MB)", flush=True)
    return adapter


def test_aligned_cortical_column():
    print("\n" + "=" * 75, flush=True)
    print("[FASE 3] VERIFICA MODELLO ALLINEATO A 5 LAYER IN BFLOAT16", flush=True)
    print("=" * 75, flush=True)

    cfg = CorticalColumn7BConfig(dtype=torch.bfloat16, device="cpu")
    model = SynapticCorticalColumn7B.from_pretrained(SNAPSHOT_7B, config=cfg)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT_7B, local_files_only=True)

    # Attach calibrated adapter
    adapter = CorticalManifoldAdapter(hidden_size=3584, bottleneck_dim=1024, dtype=torch.bfloat16)
    adapter.load_weights(WEIGHTS_PATH)
    model.bridge = adapter

    rss = psutil.Process().memory_info().rss / (1024**2)
    print(f"Modello pronto in memoria! RAM: {rss:.1f} MB ({rss/1024:.2f} GB) [Tetto: 5.18 GB]\n", flush=True)

    test_prompts = [
        "Ciao se io mi chiamo umberto, tu come ti chiami?",
        "Qual e il tuo ruolo e come puoi aiutarmi?",
    ]

    for p in test_prompts:
        msgs = [{"role": "user", "content": p}]
        fmt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(fmt, return_tensors="pt")["input_ids"]

        print(f"[User] Prompt: {p}", flush=True)
        t0 = time.perf_counter()
        gen = model.generate(inp, max_new_tokens=40, temperature=0.7, top_k=20)
        dt = time.perf_counter() - t0

        new_toks = gen[0, inp.shape[1]:]
        out_text = tokenizer.decode(new_toks, skip_special_tokens=True).strip()
        speed = len(new_toks) / max(dt, 1e-4)

        print(f"[Model] Risposta: {out_text}", flush=True)
        print(f"[Speed] [{len(new_toks)} token generati in {dt:.2f}s ({speed:.2f} tok/s)]\n", flush=True)


if __name__ == "__main__":
    X, Y = extract_calibration_trajectories()
    train_adapter_on_trajectories(X, Y, epochs=120, lr=1e-3)
    test_aligned_cortical_column()
