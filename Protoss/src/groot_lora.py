"""LoRA on the action DiT's Linear weights; all pretrained parameters frozen."""
import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankWeight(nn.Module):
    def __init__(self, weight, rank, alpha):
        super().__init__()
        if rank < 1 or alpha <= 0 or not math.isfinite(alpha):
            raise ValueError("LoRA rank and alpha must be positive")
        self.scale = alpha / rank
        self.A = nn.Parameter(weight.new_empty(rank, weight.shape[1]))
        self.B = nn.Parameter(weight.new_zeros(weight.shape[0], rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, original):
        return original + self.scale * (self.B @ self.A)


def install_lora(model, rank=8, alpha=16):
    model.requires_grad_(False)
    names = []
    for name, module in list(model.action_head.model.named_modules()):
        if isinstance(module, nn.Linear):
            parametrize.register_parametrization(module, "weight", LowRankWeight(module.weight, rank, alpha))
            names.append("action_head.model." + name)
    if not names:
        raise ValueError("No Linear modules found in GR00T action DiT")
    return names


def merge_lora(model):
    for module in list(model.modules()):
        if isinstance(module, nn.Linear) and parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


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
