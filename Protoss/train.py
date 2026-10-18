"""Train the controlled GR00T Model-FM/S/E LoRA experiments."""
import argparse
import json
import sys
from pathlib import Path


def configure_training(config, args):
    """Apply model and trainer settings to the GR00T config."""
    config.load_config_path = None
    config.data.seed = args.seed
    config.data.override_pretraining_statistics = False
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = False
    config.model.tune_vlln = False
    config.model.tune_diffusion_model = True
    config.model.load_bf16 = False
    config.model.backbone_trainable_params_fp32 = True
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
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hf-repo-id", help="Upload destination ACCOUNT/REPO (default: logged-in account/output folder name)")
    parser.add_argument("--no-hf-upload", action="store_true", help="Disable automatic upload for offline training")
    smooth = parser.add_argument_group("Model-S smoothness experiment")
    smooth.add_argument("--s-jerk-weight", type=float, default=1.0,
                        help="Multiplier for normalized jerk-only Model-S loss")
    smooth.add_argument("--s-jerk-reference", type=float, default=None,
                        help="Fixed jerk RMS in rad/s^3; omit to estimate from calibration targets")
    smooth.add_argument("--s-calibration-steps", type=int, default=10)
    smooth.add_argument("--s-dt", type=float, default=1 / 15, help="Action interval in seconds")
    smooth.add_argument("--s-gradient-diagnostics", action="store_true")
    args = parser.parse_args()
    if min(args.lora_rank, args.max_steps, args.save_steps, args.batch_size, args.gradient_accumulation) < 1 or args.lr <= 0 or args.lora_alpha <= 0 or args.workers < 0:
        parser.error("Invalid positive training/LoRA parameters")
    report = json.loads((args.styles_dir / "styles.json").read_text())
    if report.get("synthetic") or report.get("style_method") != "original" or "train" not in report["splits"]:
        parser.error("Use the common original dataset prepared with --style-method original")
    if args.variant == "S":
        if __package__:
            from .src.model_s_loss import config_from_args
        else:
            from src.model_s_loss import config_from_args
        try:
            config_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))
    args.style = args.variant
    sys.path.insert(0, str(args.groot_root.resolve()))
    return args


def prepare_training(args):
    import torch
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag

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
        from .src.training_processor_settings import load_training_modalities, processor_contract
    else:
        from src.training_processor_settings import load_training_modalities, processor_contract
    from gr00t.data.utils import parse_modality_configs
    modalities = load_training_modalities(args.base_model, EmbodimentTag.resolve(TAG).value)
    config.data.modality_configs = parse_modality_configs(modalities)
    config.model = type(config.model).from_pretrained(args.base_model)
    configure_training(config, args)
    video_indices = modalities[EmbodimentTag.resolve(TAG).value]["video"]["delta_indices"]
    print(f"Pretrained processor preserved: video_delta_indices={video_indices}, normalization=pretrained", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lora_settings = {"backend": "peft", "variant": args.variant, "rank": args.lora_rank, "alpha": args.lora_alpha,
                        "base_model": str(args.base_model.resolve()), "dataset": str(dataset),
                        "seed": args.seed, "loss": "FM",
                        "processor_policy": "pretrained-v1", "processor_sha256": processor_contract(args.base_model),
                        "video_delta_indices": video_indices, "override_pretraining_statistics": False}
    if args.variant == "S":
        from dataclasses import asdict
        if __package__:
            from .src.model_s_loss import config_from_args
        else:
            from src.model_s_loss import config_from_args
        lora_settings["loss"] = "normalized_joint_jerk_only-v3"
        lora_settings["model_s"] = asdict(config_from_args(args))
    settings_path = args.output_dir / "lora_settings.json"
    if args.resume:
        previous = json.loads(settings_path.read_text()) if settings_path.exists() else None
        # Model 1 is the former name of the same FM experiment.
        if previous is not None and previous.get("variant") == "1":
            previous["variant"] = "FM"
        if previous is not None and previous.get("backend") != "peft":
            raise ValueError("PEFT training cannot resume legacy custom-LoRA checkpoints; use a new output directory")
        if previous != lora_settings:
            raise ValueError("LoRA resume settings differ or are missing; use the original experiment settings")
    write_json(args.output_dir / "style_run.json", {"style": args.style, "base_model": str(args.base_model.resolve()),
               "dataset": str(dataset), "seed": args.seed, "loss": lora_settings["loss"],
               "selection": report, "batch_size": args.batch_size, "lr": args.lr,
               "max_steps": args.max_steps, "gradient_accumulation": args.gradient_accumulation})
    write_json(settings_path, lora_settings)

    return config, lora_settings


def main(args):
    if __package__:
        from .src.training_runtime import (
            setup_training, preprocess, update_weights, save_checkpoint, export_model,
            record_training_loss, save_training_loss_plot, save_training_distribution_plot, calibrate_s_reference,
        )
    else:
        from src.training_runtime import (
            setup_training, preprocess, update_weights, save_checkpoint, export_model,
            record_training_loss, save_training_loss_plot, save_training_distribution_plot, calibrate_s_reference,
        )

    if __package__:
        from .src.hf_model_sync import training_destination, upload_training_run
    else:
        from src.hf_model_sync import training_destination, upload_training_run
    # Fail before GPU training if setup-time authentication is missing.
    destination = training_destination(args)
    config, lora_settings = prepare_training(args)
    with setup_training(config, args, lora_settings) as training:
        model = training.model
        calibrate_s_reference(training)
        for step in range(training.step + 1, args.max_steps + 1):
            step_loss = 0.0
            step_metrics = {}
            # One optimizer step accumulates gradients from several batches.
            for microstep, batch in enumerate(training.batches(args.gradient_accumulation)):
                inputs = preprocess(batch, training)
                with training.autocast():
                    if training.smoothness is None:
                        model_output = model(**inputs)
                        loss = model_output["loss"] / args.gradient_accumulation
                    else:
                        total, metrics = training.smoothness.forward(model, inputs, diagnostics=microstep == 0)
                        loss = total / args.gradient_accumulation
                        for key, value in metrics.items():
                            # Gradient diagnostics are sampled once per optimizer step.
                            divisor = 1 if key.endswith("grad_norm") else args.gradient_accumulation
                            step_metrics[key] = step_metrics.get(key, 0.0) + value / divisor
                loss.backward()
                step_loss += loss.detach().item()

            update_weights(training)
            training.step = step
            record_training_loss(training, step_loss, **step_metrics)
            print(f"step={step}/{args.max_steps} loss={step_loss:.6f}", flush=True)
            if step_metrics:
                print("Model-S " + " ".join(f"{key}={value:.8g}" for key, value in step_metrics.items()), flush=True)
            if step % args.save_steps == 0 or step == args.max_steps:
                save_checkpoint(training)

        export_model(training)
        save_training_loss_plot(training)

    # Release the training model, optimizer and workers before loading inference models.
    del model, training
    try:
        save_training_distribution_plot(args)
    finally:
        # Preserve and publish the saved model even if the optional plot fails.
        if destination:
            upload_training_run(args.output_dir, destination)


if __name__ == "__main__":
    args = parse_args()
    main(args)
