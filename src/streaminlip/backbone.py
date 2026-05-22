"""
Gemma-3-1B text backbone with LoRA.

model_type: gemma3_text  (text-only, NOT the multimodal gemma3)
Hidden dim: 1152, Layers: 26, Heads: 4 (GQA: 1 kv-head), Vocab: 262144
~1B parameters
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType


# Verified from google/gemma-3-1b-pt config.json
GEMMA3_1B_CONFIG = {
    "model_type": "gemma3_text",
    "hidden_size": 1152,
    "intermediate_size": 6912,
    "num_hidden_layers": 26,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "vocab_size": 262144,
    "max_position_embeddings": 131072,
}


def build_backbone(
    pretrained_path: str | None = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    random_init: bool = False,
) -> nn.Module:
    if pretrained_path and not random_init:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        cfg_dict = {k: v for k, v in GEMMA3_1B_CONFIG.items() if k != "model_type"}
        cfg = AutoConfig.for_model(GEMMA3_1B_CONFIG["model_type"], **cfg_dict)
        model = AutoModelForCausalLM.from_config(cfg)

    if lora_rank == 0:
        return model   # no LoRA, caller freezes as needed

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(model, lora_cfg)
