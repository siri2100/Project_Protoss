"""Load a native GR00T model or merge a saved Protoss PEFT adapter at inference."""
import json
from pathlib import Path


def resolve_inference_paths(model_path, base_model_path=None):
    root = Path(model_path)
    manifest = root / "adapter_inference.json"
    if not manifest.is_file() and (root / "inference/adapter_inference.json").is_file():
        manifest = root / "inference/adapter_inference.json"
    if manifest.is_file():
        settings = json.loads(manifest.read_text())
        if settings.get("format") != "protoss-peft-v1":
            raise ValueError(f"Unsupported adapter manifest: {manifest}")
        base = base_model_path or settings["base_model"]
        adapter = (manifest.parent / settings["adapter"]).resolve()
        processor = (manifest.parent / settings["processor"]).resolve()
    elif (root / "adapter/adapter_config.json").is_file() and (root / "lora_settings.json").is_file():
        # Support existing PEFT runs without retraining or using their merged export.
        settings = json.loads((root / "lora_settings.json").read_text())
        if settings.get("backend") != "peft":
            raise ValueError("Only PEFT adapters are supported")
        base = base_model_path or settings["base_model"]
        adapter = root / "adapter"
        processor = root / "processor"
    else:
        if base_model_path is not None:
            raise ValueError("--base-model-path requires an adapter run or adapter manifest")
        processor = root / "processor" if (root / "processor").is_dir() and not (root / "processor_config.json").exists() else root
        return root, None, processor
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"PEFT adapter config missing: {adapter}")
    if not ((adapter / "adapter_model.safetensors").is_file() or (adapter / "adapter_model.bin").is_file()):
        raise FileNotFoundError(f"PEFT adapter weights missing: {adapter}")
    if not (processor / "processor_config.json").is_file():
        raise FileNotFoundError(f"Training processor config missing: {processor}")
    return base, adapter, processor


def load_model_for_inference(model_path, base_model_path=None, lora_mode="unmerged"):
    """Choose true pretrained baseline, disabled adapter, or active PEFT inference."""
    from transformers import AutoModel

    if lora_mode not in ("pretrained", "disabled", "unmerged", "merged"):
        raise ValueError(f"Unknown LoRA mode: {lora_mode}")
    base, adapter, processor = resolve_inference_paths(model_path, base_model_path)
    if adapter is None:
        print(f"Inference mode=pretrained; model={base}; processor={processor}", flush=True)
        return AutoModel.from_pretrained(base), processor

    if lora_mode == "pretrained":
        # Match loading the original checkpoint directly, including its processor.
        native = Path(base)
        processor = native / "processor" if (native / "processor").is_dir() and not (native / "processor_config.json").exists() else base
        print(f"Inference mode=pretrained; model={base}; processor={processor}", flush=True)
        return AutoModel.from_pretrained(base), processor

    import torch

    print(f"Inference mode={lora_mode}; pretrained={base}; processor={processor}", flush=True)
    model = AutoModel.from_pretrained(base, load_bf16=False, torch_dtype=torch.float32)
    model.float()
    if lora_mode != "disabled":
        from peft import PeftModel
        print(f"Loading PEFT adapter: {adapter}", flush=True)
        dit = PeftModel.from_pretrained(model.action_head.model, adapter, is_trainable=False)
        model.action_head.model = dit.merge_and_unload(safe_merge=True) if lora_mode == "merged" else dit
    model.requires_grad_(False)
    print(f"LoRA active={lora_mode != 'disabled'}; merged={lora_mode == 'merged'}", flush=True)
    return model, processor
