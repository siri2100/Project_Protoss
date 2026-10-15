"""CPU integration tests for the explicit loop, optimizer and native checkpoints."""
from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from transformers import Trainer, TrainerCallback, TrainingArguments

import train as entry
from src import training_runtime as runtime
from src.groot_lora import install_lora


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.model = nn.Sequential(nn.Linear(2, 1, bias=False))
        self.action_head.model[0].weight.data.fill_(0.2)
        self.calls = 0

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")

    def forward(self, inputs):
        self.calls += 1
        # Exercise restoration of model-side RNG as well as weights/optimizer.
        prediction = self.action_head.model(inputs) + torch.rand(()) * 0.01
        return {"loss": (prediction - 1).square().mean()}


class ExplicitLoopTests(unittest.TestCase):
    def make_session(self, root, resume=False, batch_size=1, accumulation=2):
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        model = TinyPolicy()
        install_lora(model, rank=1, alpha=2)
        arguments = TrainingArguments(
            output_dir=str(root), use_cpu=True, max_steps=4,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            learning_rate=0.03, lr_scheduler_type="cosine", warmup_steps=1,
            report_to="none", save_total_limit=2, disable_tqdm=True,
        )
        trainer = Trainer(model=model, args=arguments)
        trainer.create_optimizer_and_scheduler(num_training_steps=4)
        args = SimpleNamespace(output_dir=root, resume=resume, max_steps=4,
                               gradient_accumulation=accumulation, save_steps=2)
        config = SimpleNamespace(training=SimpleNamespace(bf16=False, max_grad_norm=1.0))
        checkpoint = runtime.restore_checkpoint(trainer, args)
        session = runtime.TrainingSession(trainer, MagicMock(), config, args, {}, [])
        session.dataloader = [{"inputs": torch.ones(batch_size, 2)}]
        session.iterator = iter(session.dataloader)
        if checkpoint:
            trainer._load_rng_state(checkpoint)
        trainer.optimizer.zero_grad(set_to_none=True)
        return session

    @contextmanager
    def checkpoint_callback(self):
        module = ModuleType("gr00t.experiment.utils")
        module.CheckpointFormatCallback = MagicMock()
        with patch.dict(sys.modules, {"gr00t.experiment.utils": module}):
            yield

    def run_main(self, session, stop_after_checkpoint=False):
        @contextmanager
        def setup(*args):
            yield session

        with patch.object(entry, "prepare_training", return_value=(session.config, {})), \
             patch.object(runtime, "setup_training", setup), \
             patch.object(runtime.TrainingSession, "autocast", lambda self: nullcontext()), \
             patch.object(runtime, "export_model") as export, \
             patch.object(Trainer, "train", side_effect=AssertionError("hidden loop called")), \
             self.checkpoint_callback():
            original = runtime.save_checkpoint
            if stop_after_checkpoint:
                def interrupt(training):
                    original(training)
                    raise InterruptedError("simulate interruption after checkpoint")
                with patch.object(runtime, "save_checkpoint", interrupt):
                    with self.assertRaises(InterruptedError):
                        entry.main(session.args)
            else:
                entry.main(session.args)
                export.assert_called_once_with(session)

    def test_accumulation_matches_larger_batch_and_base_stays_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            accumulated = self.make_session(root / "accumulated")
            base = accumulated.model.action_head.model.base_model.model[0].base_layer.weight.detach().clone()
            # Repeat identical samples as a larger batch; disable stochastic noise
            # so each optimizer step represents exactly the same objective.
            combined = self.make_session(root / "combined", batch_size=2, accumulation=1)
            # Both runs use identical noise to isolate accumulation mathematics.
            with patch.object(torch, "rand", return_value=torch.tensor(0.5)):
                self.run_main(accumulated)
                self.run_main(combined)
            for a, b in zip(accumulated.model.parameters(), combined.model.parameters()):
                torch.testing.assert_close(a, b)
            torch.testing.assert_close(base, accumulated.model.action_head.model.base_model.model[0].base_layer.weight)
            self.assertEqual(accumulated.model.calls, 8)
            self.assertEqual(combined.model.calls, 4)
            self.assertEqual(accumulated.step, 4)

    def test_resume_matches_uninterrupted_weights_scheduler_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            uninterrupted = self.make_session(root / "full")
            self.run_main(uninterrupted)
            expected = copy.deepcopy(uninterrupted.model.state_dict())
            expected_rng = (torch.rand(()), random.random(), np.random.rand())
            interrupted = self.make_session(root / "resumed")
            self.run_main(interrupted, stop_after_checkpoint=True)
            resumed = self.make_session(root / "resumed", resume=True)
            self.assertEqual(resumed.step, 2)
            self.run_main(resumed)
            for name, value in resumed.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            actual_rng = (torch.rand(()), random.random(), np.random.rand())
            self.assertEqual(actual_rng, expected_rng)
            self.assertEqual(resumed.trainer.lr_scheduler.state_dict(),
                             uninterrupted.trainer.lr_scheduler.state_dict())
            checkpoint = root / "resumed/checkpoint-4"
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                self.assertTrue((checkpoint / name).is_file())

    def test_checkpoint_from_original_trainer_can_resume(self):
        class StopAfterTwo(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 2:
                    control.should_save = True
                    control.should_training_stop = True

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = self.make_session(root)
            original.trainer.train_dataset = [{"inputs": torch.ones(2)}] * 8
            original.trainer.add_callback(StopAfterTwo())
            original.trainer.train()
            expected = copy.deepcopy(original.model.state_dict())
            resumed = self.make_session(root, resume=True)
            self.assertEqual(resumed.step, 2)
            for name, value in resumed.model.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            for key, state in original.trainer.optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(
                        resumed.trainer.optimizer.state_dict()["state"][key][name], value,
                    )
            self.run_main(resumed)
            self.assertEqual(resumed.step, 4)

    def test_export_preserves_trained_predictions_after_merging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self.make_session(root)
            self.run_main(session)
            inputs = torch.ones(1, 2)
            expected = session.model.action_head.model(inputs).detach()
            runtime.export_model(session)
            self.assertTrue((root / "adapter/adapter_config.json").is_file())
            self.assertTrue((root / "adapter/adapter_model.safetensors").is_file())
            adapter = torch.load(root / "lora_adapter.pt", weights_only=True)
            self.assertEqual(len(adapter["state_dict"]), 2)
            self.assertTrue(any(torch.count_nonzero(value) for name, value
                                in adapter["state_dict"].items() if ".lora_B." in name))
            reloaded = TinyPolicy()
            reloaded.load_state_dict(torch.load(root / "inference/model.pt", weights_only=True))
            torch.testing.assert_close(reloaded.action_head.model(inputs), expected)
            self.assertTrue((root / "lora_run.json").is_file())
            session.pipeline.processor.save_pretrained.assert_called_with(root / "inference")

    def test_analysis_export_reads_full_peft_checkpoint(self):
        from src.analyze_groot import export_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self.make_session(root)
            self.run_main(session)
            (root / "lora_settings.json").write_text(json.dumps({
                "backend": "peft", "rank": 1, "alpha": 2,
            }))
            (root / "processor_config.json").write_text("{}")
            inputs = torch.ones(1, 2)
            expected = session.model.action_head.model(inputs).detach()
            args = SimpleNamespace(checkpoint=root / "checkpoint-4", base_model=root,
                                   output_dir=root / "analysis")
            modules = {name: ModuleType(name) for name in ("gr00t", "gr00t.model")}
            with patch.dict(sys.modules, modules), \
                 patch("transformers.AutoModel.from_pretrained", return_value=TinyPolicy()), \
                 patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()):
                export_checkpoint(args)
            reloaded = TinyPolicy()
            reloaded.load_state_dict(torch.load(root / "analysis/model.pt", weights_only=True))
            torch.testing.assert_close(reloaded.action_head.model(inputs), expected)

    def test_missing_resume_and_empty_dataloader_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "no checkpoint"):
                self.make_session(Path(tmp), resume=True)
            session = self.make_session(Path(tmp))
            session.dataloader = []
            session.iterator = iter([])
            with self.assertRaisesRegex(ValueError, "no batches"):
                list(session.batches(1))


if __name__ == "__main__":
    unittest.main()
