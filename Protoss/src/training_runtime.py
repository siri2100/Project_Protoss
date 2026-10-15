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
            pipeline.processor.save_pretrained(args.output_dir / "processor")
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
    """Keep full PEFT model/optimizer/scheduler/RNG checkpoints for resume."""
    from gr00t.experiment.utils import CheckpointFormatCallback

    trainer = training.trainer
    trainer._save_checkpoint(training.model, trial=None)
    CheckpointFormatCallback(
        run_name=training.args.output_dir.name,
        exp_cfg_dir=training.args.output_dir / "experiment_cfg",
        processor_dir=training.args.output_dir / "processor",
    ).on_save(trainer.args, trainer.state, trainer.control)


def export_model(training):
    """Save adapters separately, then merge LoRA for the existing eval scripts."""
    import torch
    from .groot_common import write_json
    from .groot_lora import merge_lora

    model = training.model
    output = training.args.output_dir
    training.trainer.save_model()
    torch.save({"settings": training.lora_settings, "state_dict": {
        name: parameter.detach().cpu() for name, parameter in model.named_parameters()
        if parameter.requires_grad}}, output / "lora_adapter.pt")
    model.action_head.model.save_pretrained(output / "adapter")
    merge_lora(model)
    export = output / "inference"
    model.save_pretrained(export)
    training.pipeline.processor.save_pretrained(export)
    write_json(output / "lora_run.json", {
        **training.lora_settings, "targets": training.targets,
        "inference_path": str(export.resolve()),
    })
