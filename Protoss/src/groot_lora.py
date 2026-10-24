"""PEFT QKV/FFN LoRA on even-indexed action DiT blocks (0, 2, 4, ...)."""
import math
import re
from contextlib import contextmanager

from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn


LORA_TARGET_SCOPE = "action-dit-even-index-qkv-ffn-v1"
LORA_TARGET_PATTERN = re.compile(
    r"transformer_blocks\.(?P<layer>\d+)\."
    r"(?:attn1\.(?:to_q|to_k|to_v)|ff\..+)"
)


def install_lora(model, rank=8, alpha=16):
    if rank < 1 or alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("LoRA rank and alpha must be positive")
    dit = model.action_head.model
    if isinstance(dit, PeftModel):
        raise ValueError("LoRA is already installed on the action DiT")
    names = []
    for name, module in dit.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        # Zero-based block indices: 0, 2, 4, ... (cross-attention in GR00T).
        # Within those blocks, keep Q/K/V and every Linear inside the FFN.
        match = LORA_TARGET_PATTERN.fullmatch(name)

        # odd blocks
        # if match is not None and int(match["layer"]) % 2 == 1:
        #     names.append(name)
        #     print(name)

        # even blocks
        #if match is not None and int(match["layer"]) % 2 == 0:
        #     names.append(name)
        #     print(name)

        # all blocks
        if match is not None:
            names.append(name)
            print(name)

    if not names:
        raise ValueError("No Q/K/V or FFN Linear modules found in even-indexed GR00T DiT blocks")

    layers = sorted({int(LORA_TARGET_PATTERN.fullmatch(name)["layer"]) for name in names})
    print(f"LoRA DiT block indices (zero-based, even): {layers}", flush=True)

    model.requires_grad_(False)
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules="(?:" + "|".join(re.escape(name) for name in names) + ")",
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
        use_rslora=False,
    )
    model.action_head.model = get_peft_model(dit, config)

    print("\nLoRA parameter freeze check:", flush=True)
    trainable_count = frozen_count = mismatch_count = 0
    for name, param in model.named_parameters():
        is_lora = ".lora_A." in name or ".lora_B." in name
        matches = param.requires_grad == is_lora
        mismatch_count += int(not matches)
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()
    return ["action_head.model." + name for name in names]


def merge_lora(model):
    dit = model.action_head.model
    if isinstance(dit, PeftModel):
        model.action_head.model = dit.merge_and_unload(safe_merge=True)


def restore_adapter(model, adapter_path):
    """Restore adapter weights in place so optimizer references remain valid."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    dit = model.action_head.model
    config = LoraConfig.from_pretrained(adapter_path)
    current = dit.peft_config["default"]
    for key in ("r", "lora_alpha", "target_modules", "lora_dropout", "bias", "use_rslora"):
        if getattr(config, key) != getattr(current, key):
            raise ValueError(f"Resume adapter configuration differs: {key}")
    weights = load_peft_weights(str(adapter_path), device="cpu")
    expected = get_peft_model_state_dict(dit)
    if set(weights) != set(expected) or any(weights[k].shape != expected[k].shape for k in expected):
        raise ValueError("Resume adapter weights do not match the installed LoRA modules")
    set_peft_model_state_dict(dit, weights)


@contextmanager
def training_pipeline(config, rank, alpha, preserve_joint_reference=False):
    """Install adapters after base loading; preserve upstream processor/trainer."""
    from gr00t.model import MODEL_REGISTRY
    original = MODEL_REGISTRY.get(type(config.model))
    holder = {}

    class LoRAPipeline(original):
        def _create_dataset(self, save_cfg_dir):
            from .training_processor_settings import create_training_dataset
            return create_training_dataset(self, save_cfg_dir, preserve_joint_reference=preserve_joint_reference)

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
