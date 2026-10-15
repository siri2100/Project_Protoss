"""CPU integration tests for the explicit loop, optimizer and native checkpoints."""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import copy
import csv
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

from src import inference_loading as inference


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = nn.Module()
        self.action_head.model = nn.Sequential(OrderedDict(to_q=nn.Linear(2, 1, bias=False)))
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
        session = runtime.TrainingSession(trainer, MagicMock(), config, args, {"base_model": "/pretrained"}, [])
        def save_processor(directory):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "processor_config.json").write_text("{}")
        session.pipeline.processor.save_pretrained.side_effect = save_processor
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
            self.assertEqual(resumed.trainer.state.log_history,
                             uninterrupted.trainer.state.log_history)
            with (root / "resumed/training_loss.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([int(row["step"]) for row in rows], [1, 2, 3, 4])
            self.assertEqual([float(row["loss"]) for row in rows],
                             [row["loss"] for row in resumed.trainer.state.log_history])
            from PIL import Image
            with Image.open(root / "resumed/training_loss.png") as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (1350, 750))
                image.verify()
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

    def test_export_saves_only_adapter_and_merges_at_inference(self):
        from peft import PeftModel
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = self.make_session(root)
            self.run_main(session)
            inputs = torch.ones(1, 2)
            expected = session.model.action_head.model(inputs).detach()
            runtime.export_model(session)
            # Training must leave the live model unmerged and save no base weights.
            self.assertIsInstance(session.model.action_head.model, PeftModel)
            self.assertTrue((root / "adapter/adapter_config.json").is_file())
            self.assertTrue((root / "adapter/adapter_model.safetensors").is_file())
            self.assertFalse((root / "model.safetensors").exists())
            self.assertFalse((root / "inference/model.pt").exists())
            self.assertFalse((root / "checkpoint-4/model.safetensors").exists())
            self.assertTrue((root / "checkpoint-4/adapter/adapter_model.safetensors").is_file())
            with patch("transformers.AutoModel.from_pretrained", return_value=TinyPolicy()) as load:
                merged, processor = inference.load_model_for_inference(root / "inference", "/override/base", "merged")
                self.assertEqual(load.call_args.args[0], "/override/base")
            torch.testing.assert_close(merged.action_head.model(inputs), expected)
            self.assertFalse(isinstance(merged.action_head.model, PeftModel))
            self.assertTrue(all(not p.requires_grad for p in merged.parameters()))
            self.assertEqual(processor, root.resolve() / "processor")
            # Existing PEFT run outputs work without an inference manifest too.
            (root / "inference/adapter_inference.json").unlink()
            (root / "lora_settings.json").write_text(json.dumps({"backend": "peft", "base_model": "/pretrained"}))
            with patch("transformers.AutoModel.from_pretrained", return_value=TinyPolicy()):
                merged, _ = inference.load_model_for_inference(root, lora_mode="merged")
            torch.testing.assert_close(merged.action_head.model(inputs), expected)

    def test_inference_modes_distinguish_disabled_lora_and_processor_source(self):
        from peft import PeftModel
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "original"
            base.mkdir()
            (base / "processor_config.json").write_text("{}")
            session = self.make_session(root)
            self.run_main(session)
            runtime.export_model(session)
            inputs = torch.ones(1, 2)
            outputs, processors = {}, {}
            for mode in ("pretrained", "disabled", "unmerged", "merged"):
                with patch("transformers.AutoModel.from_pretrained", return_value=TinyPolicy()) as load:
                    model, processor = inference.load_model_for_inference(root, str(base), mode)
                if mode in ("pretrained", "disabled"):
                    load.assert_called_once_with(str(base))
                self.assertEqual(isinstance(model.action_head.model, PeftModel), mode == "unmerged")
                outputs[mode] = model.action_head.model(inputs).detach()
                processors[mode] = Path(processor)
            torch.testing.assert_close(outputs["pretrained"], outputs["disabled"])
            torch.testing.assert_close(outputs["unmerged"], outputs["merged"])
            self.assertFalse(torch.equal(outputs["disabled"], outputs["unmerged"]))
            self.assertEqual(processors["pretrained"], base)
            for mode in ("disabled", "unmerged", "merged"):
                self.assertEqual(processors[mode], root.resolve() / "processor")

    def test_preserved_processor_export_uses_original_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "original"
            base.mkdir()
            artifacts = {"processor_config.json": '{"processor_kwargs":{"original":true}}\n',
                         "statistics.json": '{"droid":{"action":{"mean":[1.25]}}}\n'}
            for name, content in artifacts.items():
                (base / name).write_text(content)
            session = self.make_session(root / "output")
            session.lora_settings.update(base_model=str(base), processor_policy="pretrained-v1")
            runtime.export_model(session)
            for name in artifacts:
                self.assertEqual((session.args.output_dir / "processor" / name).read_bytes(),
                                 (base / name).read_bytes())
            session.pipeline.processor.save_pretrained.assert_not_called()

    def test_native_pretrained_loading_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "processor").mkdir()
            baseline = TinyPolicy()
            with patch("transformers.AutoModel.from_pretrained", return_value=baseline) as load:
                model, processor = inference.load_model_for_inference(root)
            self.assertIs(model, baseline)
            self.assertEqual(processor, root / "processor")
            load.assert_called_once_with(root)

    def test_analysis_export_reads_adapter_checkpoint(self):
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
