"""Measure pretrained GR00T's native masked flow-matching objective."""
import argparse
from contextlib import nullcontext
import csv
from itertools import islice
import json
import math
from pathlib import Path
import sys


def measure(model, dataloader, *, batches, device, bf16=True):
    """Aggregate native masked squared errors rather than averaging unequal masks."""
    import torch

    def transfer(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: transfer(item) for key, item in value.items()}
        return value

    rows = []
    total_error = total_mask = 0.0
    samples = 0
    with torch.no_grad():
        for step, batch in enumerate(islice(dataloader, batches), start=1):
            inputs = transfer(dict(batch))
            context = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()
            with context:
                output = model(**inputs)
            loss = float(output["loss"].detach())
            error = float(output["action_loss"].detach().double().sum())
            mask = float(output["action_mask"].detach().double().sum())
            if not all(math.isfinite(value) for value in (loss, error, mask)) or mask <= 0:
                raise ValueError(f"Non-finite loss or empty action mask at batch {step}")
            samples += int(output["action_mask"].shape[0])
            total_error += error
            total_mask += mask
            rows.append({"batch": step, "loss": loss, "masked_squared_error_sum": error,
                         "valid_action_elements": mask})
            print(f"batch={step}/{batches} fm_loss={loss:.6f}", flush=True)
    if not rows:
        raise ValueError("Dataset produced no batches")
    return {"fm_loss": total_error / (total_mask + 1e-6),
            "mean_batch_loss": sum(row["loss"] for row in rows) / len(rows),
            "batches": len(rows), "samples": samples, "valid_action_elements": total_mask}, rows


def evaluate(args):
    import copy
    import torch
    import gr00t.model  # Register pretrained AutoModel/AutoProcessor classes.
    from transformers import AutoModel, AutoProcessor, set_seed
    from torch.utils.data import DataLoader
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.data.utils import parse_modality_configs
    from .dataset_paths import split_path
    from .training_processor_settings import load_training_modalities, processor_path, processor_contract

    set_seed(args.seed)
    dataset_path = split_path(args.dataset_root.resolve(), args.split)
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
        {"dataset_paths": [str(dataset_path)], "mix_ratio": 1.0, "embodiment_tag": args.tag}]}})
    config.data.seed = args.seed
    config.data.override_pretraining_statistics = False
    config.data.modality_configs = parse_modality_configs(load_training_modalities(args.base_model, args.tag))
    processor = AutoProcessor.from_pretrained(processor_path(args.base_model))
    # Actions/labels and the same augmentations as training must be retained.
    processor.train()
    original_stats = copy.deepcopy(processor.state_action_processor.statistics)
    dataset, _ = DatasetFactory(config=config).build(processor=processor)
    try:
        if processor.state_action_processor.statistics != original_stats:
            raise ValueError("Dataset initialization changed pretrained normalization")
        model = AutoModel.from_pretrained(
            args.base_model, load_bf16=False, tune_llm=False, tune_visual=False,
            tune_projector=False, tune_vlln=False, tune_diffusion_model=True,
            backbone_trainable_params_fp32=True,
        ).to(args.device)
        model.requires_grad_(False)
        model.train(args.model_mode == "train")
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0,
                            collate_fn=processor.collator)
        summary, rows = measure(model, loader, batches=args.batches, device=args.device, bf16=not args.fp32)
    finally:
        if hasattr(dataset, "close"):
            dataset.close()
    summary.update(base_model=str(args.base_model.resolve()), dataset=str(dataset_path),
                   seed=args.seed, batch_size=args.batch_size, requested_batches=args.batches,
                   model_mode=args.model_mode, processor_mode="train", precision="fp32" if args.fp32 else "bf16",
                   processor_sha256=processor_contract(args.base_model), adapter_loaded=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pretrained_fm_loss.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "pretrained_fm_loss.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Pretrained FM loss = {summary['fm_loss']:.8f}; "
          f"mean batch loss = {summary['mean_batch_loss']:.8f}; "
          f"batches={summary['batches']}, samples={summary['samples']}", flush=True)
    print(f"Results: {args.output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-mode", choices=("train", "eval"), default="train")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    args = parser.parse_args()
    if min(args.batches, args.batch_size) < 1:
        parser.error("--batches and --batch-size must be positive")
    sys.path.insert(0, str(args.groot_root.resolve()))
    evaluate(args)
