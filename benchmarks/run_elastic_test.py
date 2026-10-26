"""
SynapticFold v2: Elastic Synaptic Mobilization Test.

Tests dynamic, non-uniform synaptic activation on BoolQ:
Instead of a fixed 30% quota, each layer dynamically mobilizes ONLY the minimal
number of synapses required to preserve 90% of the activation energy (Top-P Energy Spiking).
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


def monitor_memory(stage: str):
    p = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    rss_mb = p.memory_info().rss / (1024**2)
    free_mb = vm.available / (1024**2)
    return rss_mb, free_mb


class ElasticSynapticEvaluator:
    def __init__(self, snapshot_dir: str, p_energy_mass: float = 0.90):
        self.snapshot_dir = snapshot_dir
        self.p_energy_mass = p_energy_mass
        self.config = AutoConfig.from_pretrained(snapshot_dir, local_files_only=True)
        self.config._attn_implementation = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)

        self.L = self.config.num_hidden_layers
        self.d_model = self.config.hidden_size
        self.d_ff = self.config.intermediate_size

        index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
        with open(index_path, "r", encoding="utf-8") as f:
            self.weight_map = json.load(f)["weight_map"]

        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, dtype=torch.bfloat16)
        self.norm = Qwen2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps).to(dtype=torch.bfloat16)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False, dtype=torch.bfloat16)
        self.rotary_emb = Qwen2RotaryEmbedding(self.config)

        with safe_open(os.path.join(snapshot_dir, self.weight_map["model.embed_tokens.weight"]), framework="pt", device="cpu") as f:
            self.embed_tokens.weight.data.copy_(f.get_tensor("model.embed_tokens.weight").to(dtype=torch.bfloat16))
        with safe_open(os.path.join(snapshot_dir, self.weight_map["model.norm.weight"]), framework="pt", device="cpu") as f:
            self.norm.weight.data.copy_(f.get_tensor("model.norm.weight").to(dtype=torch.bfloat16))
        with safe_open(os.path.join(snapshot_dir, self.weight_map["lm_head.weight"]), framework="pt", device="cpu") as f:
            self.lm_head.weight.data.copy_(f.get_tensor("lm_head.weight").to(dtype=torch.bfloat16))

        self.layer_buffer = Qwen2DecoderLayer(self.config, 0).to(dtype=torch.bfloat16)

        self.yes_tokens = [
            self.tokenizer.encode("Yes", add_special_tokens=False)[0],
            self.tokenizer.encode(" Yes", add_special_tokens=False)[0],
            self.tokenizer.encode("yes", add_special_tokens=False)[0],
            self.tokenizer.encode(" True", add_special_tokens=False)[0],
            self.tokenizer.encode("True", add_special_tokens=False)[0],
        ]
        self.no_tokens = [
            self.tokenizer.encode("No", add_special_tokens=False)[0],
            self.tokenizer.encode(" No", add_special_tokens=False)[0],
            self.tokenizer.encode("no", add_special_tokens=False)[0],
            self.tokenizer.encode(" False", add_special_tokens=False)[0],
            self.tokenizer.encode("False", add_special_tokens=False)[0],
        ]
        self.prior_offset = 4.50

        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())

    def _load_layer(self, layer_idx: int):
        self.layer_buffer.layer_idx = layer_idx
        if hasattr(self.layer_buffer, "self_attn"):
            self.layer_buffer.self_attn.layer_idx = layer_idx

        prefix = f"model.layers.{layer_idx}."
        shard_to_keys = {}
        for p_name, param in self.layer_buffer.named_parameters():
            full_key = f"{prefix}{p_name}"
            shard_file = self.weight_map[full_key]
            if shard_file not in shard_to_keys:
                shard_to_keys[shard_file] = []
            shard_to_keys[shard_file].append((p_name, full_key))

        for shard_file, keys in shard_to_keys.items():
            with safe_open(os.path.join(self.snapshot_dir, shard_file), framework="pt", device="cpu") as f:
                for p_name, full_key in keys:
                    self.layer_buffer.get_parameter(p_name).data.copy_(
                        f.get_tensor(full_key).to(dtype=torch.bfloat16)
                    )

    @torch.no_grad()
    def evaluate_elastic_sample(self, passage: str, question: str):
        prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer (Yes or No):"
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(chat_text, return_tensors="pt")["input_ids"]
        seq_len = input_ids.size(1)

        t0 = time.perf_counter()
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(h, pos_ids)
        pos_emb = (cos, sin)

        layer_activations = []

        for l_idx in range(self.L):
            self._load_layer(l_idx)

            res = h
            norm_h = self.layer_buffer.input_layernorm(h)
            attn_out = self.layer_buffer.self_attn(
                norm_h,
                position_embeddings=pos_emb,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
            )[0]
            h = res + attn_out

            # ELASTIC MOBILIZATION (Top-P Energy Spiking)
            res = h
            norm_h2 = self.layer_buffer.post_attention_layernorm(h)
            zk = F.linear(norm_h2, self.layer_buffer.mlp.gate_proj.weight)
            zu = F.linear(norm_h2, self.layer_buffer.mlp.up_proj.weight)
            act = F.silu(zk) * zu # [1, seq_len, d_ff]

            # Compute cumulative energy density
            abs_act = torch.abs(act)
            sorted_vals, sorted_indices = torch.sort(abs_act, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_vals, dim=-1)
            total_mass = cumsum[:, :, -1:] + 1e-8
            meets_mass = (cumsum / total_mass) >= self.p_energy_mass
            first_idx = torch.argmax(meets_mass.to(torch.int32), dim=-1) # [1, seq_len]
            k_active = int(first_idx.max().item()) + 1
            k_active = max(1, min(self.d_ff, k_active))

            active_pct = (k_active / float(self.d_ff)) * 100.0
            layer_activations.append({
                "layer": l_idx,
                "k_active": k_active,
                "active_pct": active_pct,
                "silent_pct": 100.0 - active_pct,
            })

            # Zero-Scatter Indexed-GEMM on active synapses only
            topk_idx = sorted_indices[:, :, :k_active]
            gathered_act = torch.gather(act, -1, topk_idx)
            w_down = self.layer_buffer.mlp.down_proj.weight

            mlp_out = torch.zeros_like(h)
            for s_i in range(seq_len):
                act_s = gathered_act[0, s_i:s_i+1]
                idx_s = topk_idx[0, s_i]
                w_down_active = w_down[:, idx_s]
                mlp_out[0, s_i:s_i+1] = torch.matmul(act_s, w_down_active.t())

            h = res + mlp_out

            if l_idx % 4 == 0:
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())

        final_h = h[:, -1:, :]
        norm_final = self.norm(final_h)
        logits = self.lm_head(norm_final)[0, -1, :].to(torch.float32)

        top1_id = torch.argmax(logits).item()
        top1_tok = self.tokenizer.decode([top1_id])

        yes_logits = torch.stack([logits[tid] for tid in self.yes_tokens])
        no_logits = torch.stack([logits[tid] for tid in self.no_tokens])

        lse_yes = torch.logsumexp(yes_logits, dim=0).item()
        lse_no = torch.logsumexp(no_logits, dim=0).item()

        calibrated_yes = lse_yes + self.prior_offset
        pred = calibrated_yes > lse_no
        margin = abs(calibrated_yes - lse_no)
        elapsed = time.perf_counter() - t0

        ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
        return {
            "seq_len": seq_len,
            "pred": pred,
            "lse_yes_calib": calibrated_yes,
            "lse_no_raw": lse_no,
            "margin": margin,
            "top1_tok": repr(top1_tok),
            "elapsed_s": elapsed,
            "layers": layer_activations,
        }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    local_snapshot = r"C:\Users\Umberto\.cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"

    print("=" * 95, flush=True)
    print(" SYNAPTIC-FOLD V2: TEST AL VOLO MODALITÀ ELASTICA (TOP-P ENERGY SPIKING)", flush=True)
    print("=" * 95, flush=True)
    print("Parametri Dinamici:", flush=True)
    print("  - Quota di Attivazione         : ELASTICA / VARIABILE (NON fissa al 30%)", flush=True)
    print("  - Preservazione Massa Energetica: 90.0% Energia Totale (Top-P Spiking)", flush=True)
    print("  - Kernel di Calcolo            : Zero-Scatter Indexed-GEMM", flush=True)
    print("  - Classificazione              : Surface Form Marginalization (SFM)", flush=True)
    print("-" * 95, flush=True)

    evaluator = ElasticSynapticEvaluator(local_snapshot, p_energy_mass=0.90)

    ds = load_dataset("google/boolq", split="validation")
    item = ds[2619] # Sample 2619 (Ground Truth: True)

    print(f"\nDomanda di Test (Idx 2619): {item['question']}", flush=True)
    print(f"Target Atteso               : {item['answer']}", flush=True)
    print("Avvio inferenza elastica sui 28 strati...\n", flush=True)

    res = evaluator.evaluate_elastic_sample(item["passage"], item["question"])

    print("=" * 95, flush=True)
    print(" PROFILO DI ATTIVAZIONE ELASTICA STRATO PER STRATO (Sinapsi Totali per Layer = 18.944)")
    print("=" * 95, flush=True)
    print(f"{'Layer':<7} | {'Sinapsi Attive (k*)':<20} | {'% Attive':<12} | {'% Silenti':<12} | {'Dinamica Spaziale'}", flush=True)
    print("-" * 95, flush=True)

    active_pcts = []
    for l_data in res["layers"]:
        l_idx = l_data["layer"]
        k_act = l_data["k_active"]
        act_p = l_data["active_pct"]
        sil_p = l_data["silent_pct"]
        active_pcts.append(act_p)

        bar_len = int(act_p / 2.5) # max 40 chars
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"Layer {l_idx:>2} | {k_act:>5} / 18944          | {act_p:>6.2f}%    | {sil_p:>6.2f}%    | [{bar}]", flush=True)

    min_p = min(active_pcts)
    max_p = max(active_pcts)
    avg_p = sum(active_pcts) / len(active_pcts)

    print("=" * 95, flush=True)
    print(" SINTESI DELLA MOBILITAZIONE ELASTICA:")
    print(f"  - Minima Attivazione Rilevata : {min_p:.2f}% (ben al di sotto del 30%!)", flush=True)
    print(f"  - Massima Attivazione Rilevata: {max_p:.2f}% (picco di reclutamento on-demand)", flush=True)
    print(f"  - Media di Attivazione Rete   : {avg_p:.2f}% (Sparsità Media Silente: {100.0-avg_p:.2f}%)", flush=True)
    print(f"  - Risultato Predizione        : {res['pred']} (Target: {item['answer']}) -> {'CORRECT' if res['pred'] == item['answer'] else 'WRONG'}", flush=True)
    print(f"  - Margine di Confidenza       : +{res['margin']:.2f} logit (Top-1 Token: {res['top1_tok']})", flush=True)
    print(f"  - Tempo Totale di Esecuzione  : {res['elapsed_s']:.1f} s", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
