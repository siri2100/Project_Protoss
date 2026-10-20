"""Single-GPU training helpers; the forward/backward loop lives in train.py.

GR00T supplies its dataset/collator and Transformers 4.57.3 supplies optimizer,
checkpoint and resume utilities. Trainer.train() is deliberately not called.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingSession:
    trainer: object
    pipeline: object
    config: object
    args: object
    lora_settings: dict
    targets: list
    dataloader: object = None
    iterator: object = None
    smoothness: object = None

    @property
    def model(self):
        return self.trainer.model

    @property
    def step(self):
        return self.trainer.state.global_step

    @step.setter
    def step(self, value):
        self.trainer.state.global_step = value

    def batches(self, count):
        """Restart finite datasets; max_steps counts optimizer updates, not batches."""
        for _ in range(count):
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                try:
                    yield next(self.iterator)
                except StopIteration:
                    raise ValueError("Training dataloader produced no batches") from None

    def autocast(self):
        import torch
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.config.training.bf16)


def create_trainer(pipeline, config):
    """Reuse GR00T batching and native optimizer parameter groups/scheduler."""
    from transformers import TrainingArguments
    from gr00t.experiment.trainer import Gr00tTrainer

    cfg = config.training
    arguments = TrainingArguments(
        output_dir=cfg.output_dir,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.global_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        warmup_steps=cfg.warmup_steps,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        save_only_model=False,
        bf16=cfg.bf16,
        fp16=False,
        tf32=cfg.tf32,
        optim=cfg.optim,
        dataloader_num_workers=cfg.dataloader_num_workers,
        report_to="none",
        seed=config.data.seed,
        remove_unused_columns=cfg.remove_unused_columns,
        ignore_data_skip=True,
    )
    if arguments.world_size != 1 or arguments.n_gpu != 1:
        raise ValueError("This loop requires one visible CUDA GPU; set CUDA_VISIBLE_DEVICES=0")
    trainer = Gr00tTrainer(
        model=pipeline.return_model(),
        args=arguments,
        train_dataset=pipeline.return_dataset()[0],
        data_collator=pipeline.return_collator(),
        multiprocessing_context=config.data.multiprocessing_context,
    )
    trainer.state.max_steps = cfg.max_steps
    trainer.state.logging_steps = cfg.logging_steps
    trainer.state.save_steps = cfg.save_steps
    trainer.create_optimizer_and_scheduler(num_training_steps=cfg.max_steps)
    return trainer


def restore_checkpoint(trainer, args):
    """Load native Trainer checkpoints, including those from the previous loop."""
    from transformers.trainer_callback import TrainerState
    from transformers.trainer_utils import get_last_checkpoint

    if not args.resume:
        return None
    checkpoint = get_last_checkpoint(str(args.output_dir))
    if checkpoint is None:
        raise ValueError("--resume requested but no checkpoint exists")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    for name in required:
        if not (Path(checkpoint) / name).is_file():
            raise ValueError(f"Cannot resume incomplete checkpoint: {checkpoint}/{name}")
    trainer.state = TrainerState.load_from_json(str(Path(checkpoint) / "trainer_state.json"))
    if trainer.state.global_step > args.max_steps:
        raise ValueError("Checkpoint step exceeds --max-steps")
    adapter = Path(checkpoint) / "adapter"
    if (adapter / "adapter_config.json").is_file():
        from .groot_lora import restore_adapter
        restore_adapter(trainer.model, adapter)
    else:
        # Compatibility with older full-model PEFT checkpoints.
        trainer._load_from_checkpoint(checkpoint)
    trainer._load_optimizer_and_scheduler(checkpoint)
    return checkpoint


@contextmanager
def setup_training(config, args, lora_settings):
    import os
    from transformers import set_seed
    from gr00t.model import MODEL_REGISTRY
    from gr00t.experiment.experiment import (
        _init_distributed_process_group, save_run_config_artifacts,
        save_initial_actions_artifact,
    )
    from .groot_lora import training_pipeline

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This training loop supports one GPU/process")
    _init_distributed_process_group()
    set_seed(config.data.seed)
    config.validate()
    cfg_dir = args.output_dir / "experiment_cfg"
    save_run_config_artifacts(cfg_dir, args.output_dir, config, args.output_dir.name)
    with training_pipeline(config, args.lora_rank, args.lora_alpha) as holder:
        pipeline = MODEL_REGISTRY[type(config.model)](config, cfg_dir)
        try:
            pipeline.setup()
            from .training_processor_settings import save_original_processor
            save_original_processor(args.base_model, args.output_dir / "processor")
            train_dataset, _ = pipeline.return_dataset()
            if hasattr(train_dataset, "get_initial_actions"):
                save_initial_actions_artifact(train_dataset, cfg_dir)
            trainer = create_trainer(pipeline, config)
            checkpoint = restore_checkpoint(trainer, args)
            if config.training.gradient_checkpointing:
                trainer.model.gradient_checkpointing_enable()
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            session = TrainingSession(trainer, pipeline, config, args, lora_settings, holder["targets"])
            if args.variant == "S":
                import json
                from .groot_common import TAG
                from .model_s_loss import ModelSLoss, config_from_args, joint_affine
                from gr00t.data.embodiment_tags import EmbodimentTag
                scale, offset = joint_affine(pipeline.processor, EmbodimentTag.resolve(TAG).value)
                session.smoothness = ModelSLoss(config_from_args(args), scale, offset)
                if checkpoint is not None:
                    state_path = Path(checkpoint) / "model_s_loss.json"
                    if not state_path.is_file():
                        raise ValueError("Model-S resume requires model_s_loss.json; start a new run for FM-only checkpoints")
                    session.smoothness.load_state_dict(json.loads(state_path.read_text()))
            # GR00T reseeds its dataset using global_step instead of skipping samples.
            session.dataloader = trainer.get_train_dataloader()
            session.iterator = iter(session.dataloader)
            # Worker initialization can consume RNG; restore immediately before forward.
            if checkpoint is not None:
                trainer._load_rng_state(checkpoint)
            yield session
        finally:
            for dataset in (getattr(pipeline, "train_dataset", None),
                            getattr(pipeline, "eval_dataset", None)):
                if dataset is not None and hasattr(dataset, "close"):
                    dataset.close()


def preprocess(batch, training):
    """Dataset/GR00T collator already processed images, states and actions.

    This stage only transfers that collated batch to the model's device.
    """
    return training.trainer._prepare_inputs(batch)


def calibrate_s_reference(training):
    """Use target batches only; no optimizer, scheduler or model forward calls."""
    objective = training.smoothness
    if objective is None or objective.reference is not None:
        return
    remaining = objective.config.calibration_batches - objective.calibration_count
    for batch in training.batches(remaining):
        objective.calibrate(preprocess(batch, training))


def update_weights(training):
    import torch
    trainer = training.trainer
    if training.config.training.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            (p for p in training.model.parameters() if p.requires_grad),
            training.config.training.max_grad_norm,
        )
    trainer.optimizer.step()
    trainer.lr_scheduler.step()
    trainer.optimizer.zero_grad(set_to_none=True)


def save_checkpoint(training):
    """Save adapter-only model weights plus optimizer/scheduler/RNG for resume."""
    from gr00t.experiment.utils import CheckpointFormatCallback

    trainer = training.trainer
    checkpoint = training.args.output_dir / f"checkpoint-{training.step}"
    training.model.action_head.model.save_pretrained(checkpoint / "adapter")
    trainer._save_optimizer_and_scheduler(str(checkpoint))
    trainer._save_rng_state(str(checkpoint))
    trainer.state.save_to_json(str(checkpoint / "trainer_state.json"))
    if training.smoothness is not None:
        from .groot_common import write_json
        write_json(checkpoint / "model_s_loss.json", training.smoothness.state_dict())
    CheckpointFormatCallback(
        run_name=training.args.output_dir.name,
        exp_cfg_dir=training.args.output_dir / "experiment_cfg",
        processor_dir=training.args.output_dir / "processor",
    ).on_save(trainer.args, trainer.state, trainer.control)
    trainer._rotate_checkpoints(use_mtime=False, output_dir=str(training.args.output_dir))


def record_training_loss(training, loss, **metrics):
    """Persist every optimizer-step mean loss in the resumable Trainer state."""
    training.trainer.state.log_history.append({"step": training.step, "loss": float(loss), **metrics})


def save_training_loss_plot(training):
    """Write a headless PNG and CSV, including loss restored from checkpoints."""
    import csv
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    # Ignore final summaries; keep the last entry for any repeated step.
    losses = {row["step"]: row["loss"] for row in training.trainer.state.log_history
              if "step" in row and "loss" in row and row["step"] <= training.step}
    steps = sorted(losses)
    components = {row["step"]: row for row in training.trainer.state.log_history
                  if "step" in row and "normalized_jerk_loss" in row and row["step"] <= training.step}
    model_s = training.smoothness is not None or bool(components)
    output = training.args.output_dir
    with (output / "training_loss.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        keys = ["fmd_loss", "raw_jerk_loss", "normalized_jerk_loss", "total_loss",
                "fm_weight", "weighted_fmd_loss",
                "weighted_jerk_loss", "jerk_lambda", "jerk_reference",
                "fmd_grad_norm", "jerk_grad_norm"] if model_s else []
        writer.writerow(["step", "loss", *keys])
        for step in steps:
            row = {**components.get(step, {}), "total_loss": losses[step]}
            writer.writerow([step, losses[step], *(row.get(key, "") for key in keys)])

    figure = Figure(figsize=(9, 10) if model_s else (9, 5), layout="constrained")
    if model_s:
        axes = figure.subplots(3, 1, sharex=True)
        series = [("fmd_loss", "Pretrained flow preservation MSE"),
                  ("normalized_jerk_loss", "Normalized mean squared joint jerk"),
                  ("total_loss", "Weighted flow preservation + lambda * normalized jerk")]
    else:
        axes = [figure.subplots()]
        series = [("loss", "Mean training objective")]
    for axis, (key, label) in zip(axes, series):
        values = losses if key in ("loss", "total_loss") else {
            step: row[key] for step, row in components.items() if key in row}
        recorded_steps = sorted(values)
        if recorded_steps:
            axis.plot(recorded_steps, [values[step] for step in recorded_steps], linewidth=1.2)
        else:
            axis.text(0.5, 0.5, "No recorded losses in this checkpoint",
                      ha="center", va="center", transform=axis.transAxes)
        axis.set(title=key if model_s else "Training loss", ylabel=label)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Optimizer step")
    figure.set_dpi(150)
    path = output / "training_loss.png"
    FigureCanvasAgg(figure).print_png(path)
    print(f"Training loss plot: {path}", flush=True)


def export_model(training):
    """Save the PEFT adapter and processor; merge only when inference loads them."""
    from .groot_common import write_json

    output = training.args.output_dir
    if training.smoothness is not None:
        write_json(output / "model_s_loss.json", training.smoothness.state_dict())
    training.model.action_head.model.save_pretrained(output / "adapter")
    if training.lora_settings.get("processor_policy") == "pretrained-v1":
        from .training_processor_settings import save_original_processor
        save_original_processor(training.lora_settings["base_model"], output / "processor")
    else:
        training.pipeline.processor.save_pretrained(output / "processor")
    write_json(output / "inference/adapter_inference.json", {
        "format": "protoss-peft-v1",
        "base_model": training.lora_settings["base_model"],
        "adapter": "../adapter",
        "processor": "../processor",
    })
    write_json(output / "lora_run.json", {
        **training.lora_settings, "targets": training.targets,
        "inference_path": str((output / "inference").resolve()),
    })


def save_training_distribution_plot(args):
    """Compare the final saved adapter with pretrained after training resources release."""
    import gc
    from argparse import Namespace
    import torch
    from .action_distribution import compare
    from .gr00t_adapter_patch import ensure_adapter_inference

    gc.collect()
    torch.cuda.empty_cache()
    ensure_adapter_inference(args.groot_root)
    print("Training complete; generating action distribution plot...", flush=True)
    return compare(Namespace(
        groot_root=args.groot_root, base_model=args.base_model,
        model_path=args.output_dir, dataset_root=args.styles_dir,
        output_dir=args.output_dir, split="train", traj_ids=[0],
        frames_per_trajectory=32, bins=40, seed=args.seed,
        denoising_steps=4, lora_mode="unmerged", device="cuda:0",
        tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
    ))
