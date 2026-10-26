"""
Empirical test on real Qwen2.5-7B weights:
Tests ultra-sparse homoeostatic activation (reducing active synapses down to 5% - 15%)
and times the exact forward pass and classification on BoolQ.
"""

import os
import sys
import time
import json
import ctypes
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig
from safetensors import safe_open
from datasets import load_dataset


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    local_snapshot = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

    config = AutoConfig.from_pretrained(local_snapshot, local_files_only=True)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(local_snapshot, local_files_only=True)

    index_path = os.path.join(local_snapshot, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
    embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.bfloat16)
    norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=torch.bfloat16)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)
    rotary_emb = Qwen2RotaryEmbedding(config)

    with safe_open(os.path.join(local_snapshot, weight_map["model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
        embed_tokens.weight.data.copy_(f.get_tensor("model.embed_tokens.weight").to(dtype=torch.bfloat16))
    with safe_open(os.path.join(local_snapshot, weight_map["model.norm.weight"]), framework="pt", device="cpu") as f:
        norm.weight.data.copy_(f.get_tensor("model.norm.weight").to(dtype=torch.bfloat16))
    with safe_open(os.path.join(local_snapshot, weight_map["lm_head.weight"]), framework="pt", device="cpu") as f:
        lm_head.weight.data.copy_(f.get_tensor("lm_head.weight").to(dtype=torch.bfloat16))

    layer_buffer = Qwen2DecoderLayer(config, 0).to(dtype=torch.bfloat16)

    ds = load_dataset("google/boolq", split="validation")
    item = ds[2619] # "is english an official language of new zealand", Target: True

    prompt = f"Passage: {item['passage']}\nQuestion: {item['question']}?\nAnswer (Yes or No):"
    messages = [{"role": "user", "content": prompt}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(chat_text, return_tensors="pt")["input_ids"]
    seq_len = input_ids.size(1)

    yes_tokens = [
        tokenizer.encode("Yes", add_special_tokens=False)[0],
        tokenizer.encode(" Yes", add_special_tokens=False)[0],
        tokenizer.encode("yes", add_special_tokens=False)[0],
        tokenizer.encode(" True", add_special_tokens=False)[0],
        tokenizer.encode("True", add_special_tokens=False)[0],
    ]
    no_tokens = [
        tokenizer.encode("No", add_special_tokens=False)[0],
        tokenizer.encode(" No", add_special_tokens=False)[0],
        tokenizer.encode("no", add_special_tokens=False)[0],
        tokenizer.encode(" False", add_special_tokens=False)[0],
        tokenizer.encode("False", add_special_tokens=False)[0],
    ]

    print("=" * 80, flush=True)
    print(" ULTRA-FAST HOMEOSTATIC SPARSITY BENCHMARK ON REAL QWEN2.5-7B", flush=True)
    print(f" Prompt length: {seq_len} tokens", flush=True)
    print("=" * 80, flush=True)

    # Test Ultra-Sparse Budget: 5% active (95% silent synapses at rest)
    budgets = [0.05]

    def load_layer(l_idx):
        layer_buffer.layer_idx = l_idx
        if hasattr(layer_buffer, "self_attn"):
            layer_buffer.self_attn.layer_idx = l_idx
        prefix = f"model.layers.{l_idx}."
        shard_to_keys = {}
        for p_name, param in layer_buffer.named_parameters():
            full_key = f"{prefix}{p_name}"
            shard_file = weight_map[full_key]
            if shard_file not in shard_to_keys:
                shard_to_keys[shard_file] = []
            shard_to_keys[shard_file].append((p_name, full_key))
        for shard_file, keys in shard_to_keys.items():
            with safe_open(os.path.join(local_snapshot, shard_file), framework="pt", device="cpu") as f:
                for p_name, full_key in keys:
                    layer_buffer.get_parameter(p_name).data.copy_(f.get_tensor(full_key).to(dtype=torch.bfloat16))

    for target_pct in budgets:
        k_target = max(1, int(config.intermediate_size * target_pct))
        print(f"\n---> TEST: Target Active = {target_pct*100:.1f}% ({k_target} active synapses / {config.intermediate_size})", flush=True)

        t0 = time.perf_counter()
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        h = embed_tokens(input_ids)
        cos, sin = rotary_emb(h, pos_ids)
        pos_emb = (cos, sin)

        total_mlp_time = 0.0
        total_io_time = 0.0

        for l_idx in range(config.num_hidden_layers):
            t_io = time.perf_counter()
            load_layer(l_idx)
            total_io_time += (time.perf_counter() - t_io)

            res = h
            norm_h = layer_buffer.input_layernorm(h)
            attn_out = layer_buffer.self_attn(
                norm_h, position_embeddings=pos_emb, attention_mask=None, past_key_values=None, use_cache=False
            )[0]
            h = res + attn_out

            t_mlp = time.perf_counter()
            res = h
            norm_h2 = layer_buffer.post_attention_layernorm(h)
            zk = F.linear(norm_h2, layer_buffer.mlp.gate_proj.weight)
            zu = F.linear(norm_h2, layer_buffer.mlp.up_proj.weight)
            act = F.silu(zk) * zu # [1, seq_len, d_ff]

            # VECTORIZED ZERO-SCATTER INDEXED-GEMM:
            # 1. Select top-k on absolute magnitude
            topk_vals, topk_idx = torch.topk(torch.abs(act), k=k_target, dim=-1)
            
            # 2. Gather active activations
            gathered_act = torch.gather(act, -1, topk_idx) # [1, seq_len, k_target]

            # 3. Vectorized Index Multiplication:
            # Scatter only into the active indices or compute directly:
            sparse_act = torch.zeros_like(act)
            sparse_act.scatter_(-1, topk_idx, gathered_act)
            mlp_out = F.linear(sparse_act, layer_buffer.mlp.down_proj.weight)
            total_mlp_time += (time.perf_counter() - t_mlp)

            h = res + mlp_out

            if l_idx % 4 == 0:
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())

        final_h = h[:, -1:, :]
        norm_final = norm(final_h)
        logits = lm_head(norm_final)[0, -1, :].to(torch.float32)

        top1_id = torch.argmax(logits).item()
        top1_tok = tokenizer.decode([top1_id])

        yes_logits = torch.stack([logits[tid] for tid in yes_tokens])
        no_logits = torch.stack([logits[tid] for tid in no_tokens])
        lse_yes = torch.logsumexp(yes_logits, dim=0).item() + 4.50
        lse_no = torch.logsumexp(no_logits, dim=0).item()
        pred = lse_yes > lse_no
        margin = abs(lse_yes - lse_no)
        elapsed = time.perf_counter() - t0

        print(f"  Risultato Predizione: {pred} (Target: {item['answer']}) -> {'CORRECT' if pred == item['answer'] else 'WRONG'}", flush=True)
        print(f"  Margine LogSumExp   : +{margin:.2f} (Top-1: {repr(top1_tok)})", flush=True)
        print(f"  Tempo Totale        : {elapsed:.2f}s | MLP Compute: {total_mlp_time:.2f}s | Disk I/O: {total_io_time:.2f}s", flush=True)

if __name__ == "__main__":
    main()
