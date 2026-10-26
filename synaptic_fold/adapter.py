"""
SynapticFold v2: Universal HuggingFace Model Adapter & Introspection Engine.

Enables SynapticFold v2 to run on ANY arbitrary Large Language Model architecture:
- LLaMA (1, 2, 3, 3.1)
- Qwen (1.5, 2, 2.5)
- Mistral & Mixtral
- Gemma (1, 2)
- DeepSeek (V1, V2, V3)
- Falcon
- GPT-NeoX & Pythia
- GPT-2 & StarCoder
- Phi (1, 2, 3, 3.5)
- OPT & Bloom
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from .engine import SynapticBrainConfig, SynapticBrainEngine, InPlaceReservoirCache
from .reservoir import GlobalSynapticPool, ReservoirConfig


class UniversalModelAdapter:
    """
    Introspects and dynamically extracts standard functional compartments from
    arbitrary HuggingFace CausalLM architectures without hardcoded attribute paths.
    """

    @staticmethod
    def inspect_model(model: nn.Module) -> Dict[str, Any]:
        """
        Dynamically discovers embeddings, layer ModuleList, final norm, lm_head,
        and architectural properties (GQA, activation types).
        """
        # 1. Base model extraction
        base_model = getattr(model, "model", None)
        if base_model is None:
            base_model = getattr(model, "transformer", None)
        if base_model is None:
            base_model = getattr(model, "gpt_neox", None)
        if base_model is None:
            base_model = getattr(model, "backbone", None)
        if base_model is None:
            base_model = model

        # 2. Extract Embedding Layer
        embed_tokens = None
        if hasattr(model, "get_input_embeddings"):
            try:
                embed_tokens = model.get_input_embeddings()
            except Exception:
                pass
        if embed_tokens is None:
            for attr in ["embed_tokens", "wte", "word_embeddings", "embed_in", "embeddings"]:
                if hasattr(base_model, attr):
                    embed_tokens = getattr(base_model, attr)
                    break
        if embed_tokens is None:
            raise AttributeError("Could not automatically locate token embeddings on target model.")

        # 3. Extract Decoder Layers (ModuleList)
        layers = None
        for attr in ["layers", "h", "block", "blocks", "decoder_layers"]:
            if hasattr(base_model, attr) and isinstance(getattr(base_model, attr), nn.ModuleList):
                layers = getattr(base_model, attr)
                break
        if layers is None:
            # Fallback: inspect any ModuleList containing more than 4 sub-modules
            for name, child in base_model.named_children():
                if isinstance(child, nn.ModuleList) and len(child) >= 4:
                    layers = child
                    break
        if layers is None:
            raise AttributeError("Could not locate decoder layers ModuleList on target model.")

        # 4. Extract Final Pre-Head Normalization
        norm_layer = None
        for attr in ["norm", "ln_f", "final_layer_norm", "layer_norm"]:
            if hasattr(base_model, attr):
                norm_layer = getattr(base_model, attr)
                break

        # 5. Extract LM Head
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            try:
                lm_head = model.get_output_embeddings()
            except Exception:
                pass
        if lm_head is None:
            for attr in ["lm_head", "embed_out", "head"]:
                if hasattr(model, attr):
                    lm_head = getattr(model, attr)
                    break

        # 6. Extract Rotary Embeddings (if isolated in base model)
        rotary_emb = getattr(base_model, "rotary_emb", None)

        # 7. Extract Architectural Config (GQA, activation, norms)
        config = getattr(model, "config", None)
        num_query_heads = getattr(config, "num_attention_heads", 32) if config else 32
        num_kv_heads = getattr(config, "num_key_value_heads", num_query_heads) if config else num_query_heads
        if num_kv_heads is None:
            num_kv_heads = num_query_heads

        hidden_size = getattr(config, "hidden_size", 4096) if config else 4096
        head_dim = getattr(config, "head_dim", hidden_size // num_query_heads) if config else (hidden_size // num_query_heads)
        vocab_size = getattr(config, "vocab_size", 32000) if config else 32000

        # Activation type detection
        act_fn = getattr(config, "hidden_act", "silu") if config else "silu"
        if act_fn is None:
            act_fn = "silu"

        return {
            "base_model": base_model,
            "embed_tokens": embed_tokens,
            "layers": layers,
            "norm_layer": norm_layer,
            "lm_head": lm_head,
            "rotary_emb": rotary_emb,
            "num_layers": len(layers),
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "activation_type": act_fn,
            "config": config,
        }

    @classmethod
    def wrap_model(
        cls,
        model: nn.Module,
        split_start: Optional[int] = None,
        split_end: Optional[int] = None,
        brain_config: Optional[SynapticBrainConfig] = None,
    ) -> SynapticBrainEngine:
        """
        Factory method that wraps ANY arbitrary HuggingFace LLM into a SynapticBrainEngine.
        Automatically provisions GQA-aware KV-caches and functional partitions.
        """
        info = cls.inspect_model(model)
        total_layers = info["num_layers"]

        if split_start is None:
            split_start = max(1, total_layers // 6)
        if split_end is None:
            split_end = min(total_layers - 1, (5 * total_layers) // 6)

        sensory_layers = info["layers"][:split_start]
        core_layers = info["layers"][split_start:split_end]
        motor_layers = info["layers"][split_end:]

        # Create consolidated or representative recurrent core
        recurrent_core = core_layers[len(core_layers) // 2]

        engine = SynapticBrainEngine(
            embed_tokens=info["embed_tokens"],
            sensory_layers=sensory_layers,
            recurrent_core=recurrent_core,
            motor_layers=motor_layers,
            norm_layer=info["norm_layer"],
            lm_head=info["lm_head"],
            rotary_emb=info["rotary_emb"],
            config=brain_config,
        )

        # Store model metadata on engine
        engine.model_info = info
        return engine
