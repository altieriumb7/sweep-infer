import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import ctypes

def main():
    local_snapshot = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
    index_path = os.path.join(local_snapshot, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_map = json.load(f)["weight_map"]

    config = AutoConfig.from_pretrained(local_snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(local_snapshot, local_files_only=True)

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

    print("=" * 85, flush=True)
    print(" SYNAPTIC-FOLD V2: PARETO FRONTIER SEARCH ON REAL QWEN2.5-7B", flush=True)
    print(f" Target Question: {item['question']} | Expected Answer: {item['answer']}", flush=True)
    print(f" Prompt length: {seq_len} tokens | Total Synapses per Layer: {config.intermediate_size}", flush=True)
    print("=" * 85, flush=True)

    # Candidate Sparsity Regimes:
    # 1. 10.0% active (1894 synapses)
    # 2. 5.0% active  (947 synapses)
    # 3. 3.0% active  (568 synapses)
    # 4. 2.0% active  (378 synapses)
    # 5. 1.0% active  (189 synapses)
    # 6. Elastic Top-P (Target 85% energy mass dynamic breathing)
    configurations = {
        "10.0% Active": {"type": "fixed", "pct": 0.10},
        "5.0% Active":  {"type": "fixed", "pct": 0.05},
        "3.0% Active":  {"type": "fixed", "pct": 0.03},
        "2.0% Active":  {"type": "fixed", "pct": 0.02},
        "1.0% Active":  {"type": "fixed", "pct": 0.01},
        "Elastic-85%":  {"type": "elastic", "p_mass": 0.85},
    }

    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    h_init = embed_tokens(input_ids)
    cos, sin = rotary_emb(h_init, pos_ids)
    pos_emb = (cos, sin)

    states = {name: h_init.clone() for name in configurations}
    active_synapse_history = {name: [] for name in configurations}
    mlp_times = {name: 0.0 for name in configurations}

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

    t_start = time.perf_counter()
    total_io_time = 0.0

    print("\n[+] Stepping through 28 Transformer Layers (Single Pass Disk Streaming)...", flush=True)

    for l_idx in range(config.num_hidden_layers):
        t_io = time.perf_counter()
        load_layer(l_idx)
        total_io_time += (time.perf_counter() - t_io)

        for name, cfg in configurations.items():
            h_curr = states[name]
            # 1. Self Attention
            res = h_curr
            norm_h = layer_buffer.input_layernorm(h_curr)
            attn_out = layer_buffer.self_attn(
                norm_h, position_embeddings=pos_emb, attention_mask=None, past_key_values=None, use_cache=False
            )[0]
            h_curr = res + attn_out

            # 2. MLP with Sparsity Selection
            t_m = time.perf_counter()
            res = h_curr
            norm_h2 = layer_buffer.post_attention_layernorm(h_curr)
            zk = F.linear(norm_h2, layer_buffer.mlp.gate_proj.weight)
            zu = F.linear(norm_h2, layer_buffer.mlp.up_proj.weight)
            act = F.silu(zk) * zu
            abs_act = torch.abs(act)

            if cfg["type"] == "fixed":
                k_val = max(1, int(config.intermediate_size * cfg["pct"]))
                topk_vals, topk_idx = torch.topk(abs_act, k=k_val, dim=-1)
                active_count = k_val
            else: # elastic top-p energy mass
                sorted_vals, sorted_indices = torch.sort(abs_act, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_vals, dim=-1)
                total_mass = cumsum[:, :, -1:] + 1e-8
                energy_fraction = cumsum / total_mass
                meets_mass = (energy_fraction >= cfg["p_mass"])
                first_idx = torch.argmax(meets_mass.to(torch.int32), dim=-1)
                k_val = int(first_idx.max().item()) + 1
                k_val = max(1, min(config.intermediate_size, k_val))
                topk_idx = sorted_indices[:, :, :k_val]
                active_count = k_val

            active_synapse_history[name].append(active_count)

            # Vectorized Scatter + Linear Projection
            gathered_act = torch.gather(act, -1, topk_idx)
            sparse_act = torch.zeros_like(act)
            sparse_act.scatter_(-1, topk_idx, gathered_act)
            mlp_out = F.linear(sparse_act, layer_buffer.mlp.down_proj.weight)

            mlp_times[name] += (time.perf_counter() - t_m)
            states[name] = res + mlp_out

        if (l_idx + 1) % 7 == 0 or l_idx == config.num_hidden_layers - 1:
            print(f"  Processed Layer {l_idx+1:02d}/28...", flush=True)

        if l_idx % 4 == 0:
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())

    total_wall_time = time.perf_counter() - t_start

    print("\n" + "=" * 85, flush=True)
    print(" EMPIRICAL RESULTS & PARETO FRONTIER ANALYSIS", flush=True)
    print("=" * 85, flush=True)
    print(f"Total Wall Clock Time: {total_wall_time:.2f}s | Single-Pass Disk I/O: {total_io_time:.2f}s\n", flush=True)

    header = f"{'Regime':<15} | {'Avg Active Syn':<14} | {'Min-Max Syn':<13} | {'Pred':<6} | {'Status':<7} | {'Margin':<8} | {'Top-1 Tok':<10} | {'MLP Time':<8}"
    print(header)
    print("-" * len(header))

    results_summary = []

    for name, cfg in configurations.items():
        h_final = states[name][:, -1:, :]
        norm_final = norm(h_final)
        logits = lm_head(norm_final)[0, -1, :].to(torch.float32)

        top1_id = torch.argmax(logits).item()
        top1_tok = tokenizer.decode([top1_id])

        yes_logits = torch.stack([logits[tid] for tid in yes_tokens])
        no_logits = torch.stack([logits[tid] for tid in no_tokens])
        lse_yes = torch.logsumexp(yes_logits, dim=0).item() + 4.50
        lse_no = torch.logsumexp(no_logits, dim=0).item()
        pred = (lse_yes > lse_no)
        margin = lse_yes - lse_no
        status = "CORRECT" if pred == item["answer"] else "WRONG"

        history = active_synapse_history[name]
        avg_act = sum(history) / len(history)
        min_act = min(history)
        max_act = max(history)
        avg_pct = (avg_act / config.intermediate_size) * 100.0

        row = f"{name:<15} | {avg_act:>6.0f} ({avg_pct:>4.1f}%) | {min_act:>5d}-{max_act:<5d} | {str(pred):<6} | {status:<7} | {margin:>+7.2f} | {repr(top1_tok):<10} | {mlp_times[name]:>6.2f}s"
        print(row, flush=True)

        results_summary.append({
            "regime": name,
            "avg_active_synapses": avg_act,
            "avg_pct": avg_pct,
            "min_synapses": min_act,
            "max_synapses": max_act,
            "pred": bool(pred),
            "correct": (pred == item["answer"]),
            "margin": margin,
            "top1_token": top1_tok,
            "mlp_time_s": mlp_times[name],
        })

    with open("pareto_frontier_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n[+] Benchmark completed and results saved to pareto_frontier_results.json", flush=True)

if __name__ == "__main__":
    main()
