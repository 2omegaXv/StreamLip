"""
LM Backbone V2: SmolLM2-360M in standard input_ids mode (text-only).

Unlike the original backbone.py (which fed visual features as inputs_embeds),
this module processes only text tokens, keeping the language prior pure.

Returns:
  h_lm  (B, T, 960)    — last hidden states h̃_t^LM, used as FM condition
  s_lm  (B, T, 49152)  — LM logits log p(x_t | x_{1:t-1})
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

try:
    from peft import get_peft_model, LoraConfig, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class LMBackbone(nn.Module):
    """SmolLM2-360M wrapper: input_ids → (h_lm, s_lm)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model  # LlamaForCausalLM

    def forward(
        self,
        input_ids:      torch.Tensor,               # (B, T) int64
        attention_mask: torch.Tensor | None = None,  # (B, T) long/bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          h_lm  (B, T, 960)
          s_lm  (B, T, 49152)
        """
        # output_hidden_states=True works regardless of PEFT wrapping
        out  = self.model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
        h_lm = out.hidden_states[-1]   # (B, T, 960)  last transformer layer output
        s_lm = out.logits              # (B, T, 49152) already through lm_head
        return h_lm, s_lm


def build_lm_backbone(
    pretrained_path: str,
    lora_rank:       int = 16,
    lora_alpha:      int = 32,
) -> LMBackbone:
    """
    Load SmolLM2-360M and optionally wrap with LoRA.
    lora_rank=0 → full fine-tune (all parameters trainable).
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch.bfloat16,
        )

    if lora_rank > 0:
        assert _PEFT_AVAILABLE, "peft not installed; pass lora_rank=0 for full fine-tune"
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, cfg)

    return LMBackbone(model)
