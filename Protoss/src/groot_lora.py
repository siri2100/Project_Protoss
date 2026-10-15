"""PEFT LoRA on the action DiT; all pretrained parameters stay frozen."""
import math
import re
from contextlib import contextmanager

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn


def install_lora(model, rank=8, alpha=16):
    if rank < 1 or alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("LoRA rank and alpha must be positive")
    dit = model.action_head.model
    if isinstance(dit, PeftModel):
        raise ValueError("LoRA is already installed on the action DiT")
    names = []
    for name, param in dit.named_modules():
        if not name or not isinstance(param, nn.Linear):
            continue
        # Add module-name filters here, e.g. exclude timestep_encoder:
        # if "timestep_encoder" in name:
        #     continue
        names.append(name)
    if not names:
        raise ValueError("No Linear modules found in GR00T action DiT")

    model.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        # Match exact paths, including output and timestep projections.
        target_modules="(?:" + "|".join(re.escape(name) for name in names) + ")",
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
        use_rslora=False,
    )
    model.action_head.model = get_peft_model(dit, config)
    return ["action_head.model." + name for name in names]


def merge_lora(model):
    dit = model.action_head.model
    if isinstance(dit, PeftModel):
        model.action_head.model = dit.merge_and_unload(safe_merge=True)


@contextmanager
def training_pipeline(config, rank, alpha):
    """Install adapters after base loading; preserve upstream processor/trainer."""
    from gr00t.model import MODEL_REGISTRY
    original = MODEL_REGISTRY.get(type(config.model))
    holder = {}

    class LoRAPipeline(original):
        def _create_model(self):
            model = super()._create_model()
            holder["targets"] = install_lora(model, rank, alpha)
            holder["model"] = model
            holder["pipeline"] = self
            print(f"LoRA targets: {len(holder['targets'])}; trainable parameters: "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
            return model

    MODEL_REGISTRY[type(config.model)] = LoRAPipeline
    try:
        yield holder
    finally:
        MODEL_REGISTRY[type(config.model)] = original
