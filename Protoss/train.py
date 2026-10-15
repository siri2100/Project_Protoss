"""Train the controlled GR00T Model-FM/S/E LoRA experiments."""
import argparse
import json
import sys
from pathlib import Path


def configure_training(config, args):
    """Apply model and trainer settings to the GR00T config."""
    config.load_config_path = None
    config.data.seed = args.seed
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = str(args.base_model.resolve())
    config.training.output_dir = str(args.output_dir.resolve())
    # Upstream appends experiment_name to output_dir when it is non-None.
    config.training.experiment_name = None
    config.training.num_gpus = 1
    config.training.global_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation
    config.training.learning_rate = args.lr
    config.training.max_steps = args.max_steps
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = 3
    config.training.dataloader_num_workers = args.workers
    config.training.optim = "adamw_torch"
    config.training.use_wandb = False
    config.training.resume_from_checkpoint = args.resume


def parse_args():
    """Parse and validate training command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="variant", choices=["FM", "S", "E"], required=True)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--dataset-root", dest="styles_dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=float, default=128)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--state-dropout-prob", type=float, default=0.0,
                        help="Probability of dropping the whole state during training; default off")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.lora_rank, args.max_steps, args.save_steps, args.batch_size, args.gradient_accumulation) < 1 or args.lr <= 0 or args.lora_alpha <= 0 or args.workers < 0:
        parser.error("Invalid positive training/LoRA parameters")
    if not 0 <= args.state_dropout_prob <= 1:
        parser.error("state-dropout-prob must be between 0 and 1")
    report = json.loads((args.styles_dir / "styles.json").read_text())
    if report.get("synthetic") or report.get("style_method") != "original" or "train" not in report["splits"]:
        parser.error("Use the common original dataset prepared with --style-method original")
    args.style = args.variant
    sys.path.insert(0, str(args.groot_root.resolve()))
    return args


def main(args):
    import torch
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    if __package__:
        from .src.groot_common import TAG, write_json
        from .src.dataset_paths import split_path, require_unsplit
    else:
        from src.groot_common import TAG, write_json
        from src.dataset_paths import split_path, require_unsplit

    report = json.loads((args.styles_dir / "styles.json").read_text())
    require_unsplit(report)
    split = "train"
    dataset = split_path(args.styles_dir.resolve(), split)
    if not report["splits"][split] or not (dataset / "meta/relative_stats.json").exists():
        raise ValueError("Run prepare first; style dataset/statistics are missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty; choose a new path or --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("GR00T fine-tuning requires a CUDA GPU")
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
        {"dataset_paths": [str(dataset)], "mix_ratio": 1.0,
         "embodiment_tag": EmbodimentTag.resolve(TAG).value}]}})

    if __package__:
        from .src.training_processor_settings import load_training_modalities
    else:
        from src.training_processor_settings import load_training_modalities
    from gr00t.data.utils import parse_modality_configs
    config.data.modality_configs = parse_modality_configs(
        load_training_modalities(args.base_model, EmbodimentTag.resolve(TAG).value))
    config.model.state_dropout_prob = args.state_dropout_prob
    print(f"v1.3 processor: video_delta_indices=[0], state_dropout_prob={args.state_dropout_prob}", flush=True)
    configure_training(config, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "style_run.json", {"style": args.style, "base_model": str(args.base_model.resolve()),
               "dataset": str(dataset), "seed": args.seed, "loss": "native GR00T masked flow-matching MSE",
               "selection": report, "batch_size": args.batch_size, "lr": args.lr,
               "max_steps": args.max_steps, "gradient_accumulation": args.gradient_accumulation})

    if __package__:
        from .src.groot_lora import training_pipeline, merge_lora
    else:
        from src.groot_lora import training_pipeline, merge_lora
    config.model.tune_projector = False
    lora_settings = {"variant": args.variant, "rank": args.lora_rank, "alpha": args.lora_alpha,
                        "base_model": str(args.base_model.resolve()), "dataset": str(dataset),
                        "seed": args.seed, "loss": "FM",
                        "video_delta_indices": [0], "state_dropout_prob": args.state_dropout_prob}
    settings_path = args.output_dir / "lora_settings.json"
    if args.resume:
        previous = json.loads(settings_path.read_text()) if settings_path.exists() else None
        # Model 1 is the former name of the same FM experiment.
        if previous is not None and previous.get("variant") == "1":
            previous["variant"] = "FM"
        if previous != lora_settings:
            raise ValueError("LoRA resume settings differ or are missing; use the original experiment settings")
    write_json(settings_path, lora_settings)

    with training_pipeline(config, args.lora_rank, args.lora_alpha) as holder:
        run(config)
        torch.save({"settings": lora_settings, "state_dict": {
            name: parameter.detach().cpu() for name, parameter in holder["model"].named_parameters()
            if parameter.requires_grad}}, args.output_dir / "lora_adapter.pt")
        # Trainer checkpoints contain parametrized weights + optimizer for
        # resume. Export merged weights with native names for GR00T eval.
        merge_lora(holder["model"])
        export = args.output_dir / "inference"
        holder["model"].save_pretrained(export)
        holder["pipeline"].processor.save_pretrained(export)
        write_json(args.output_dir / "lora_run.json", {
            "variant": args.variant, "rank": args.lora_rank, "alpha": args.lora_alpha,
            "targets": holder["targets"], "base_model": str(args.base_model.resolve()),
            "dataset": str(dataset), "seed": args.seed, "loss": "FM",
            "inference_path": str(export.resolve())})


if __name__ == "__main__":
    args = parse_args()
    main(args)
