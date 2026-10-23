"""Verify README recipes with an actual isolated Python child and saved settings."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src import experiment_recipes as experiments
from src import setup_recipes as setup
from src.recipe_runtime import Recipe
from src.run_recipes import COMMANDS, main
from src.robolab_testset import TEST_TASKS


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "Project Protoss2"
        self.root.mkdir()
        config_path = Path(__file__).resolve().parents[2] / "run_config.py"
        spec = importlib.util.spec_from_file_location("test_settings", config_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config = SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(module).items() if k.isupper()})
        self.config.ROBOLAB_STACK_ROOT = str(self.root / "simulator")
        self.config.HF_ACCOUNT = "test-owner"
        self.args = SimpleNamespace(project_root=self.root, dry_run=False, repo_id=None,
                                    name=None, port=None, stack=None, result_name=None)
        self.ctx = Recipe(self.args, self.config)
        self.ctx.tools.mkdir()
        self.ctx.python.parent.mkdir(parents=True)
        self.ctx.python.symlink_to(sys.executable)
        (self.ctx.tools / "train.py").write_text('''
import json, os, pathlib, sys
path = pathlib.Path(os.environ["PROTOSS_ROOT"]) / "captured.json"
path.write_text(json.dumps({"args": sys.argv[1:], "root": os.environ["PROTOSS_ROOT"],
    "hf_transfer": os.environ["HF_HUB_ENABLE_HF_TRANSFER"],
    "venv": os.environ.get("VIRTUAL_ENV"), "uv_env": os.environ.get("UV_PROJECT_ENVIRONMENT")}))
''')

    def captured(self):
        return json.loads((self.root / "captured.json").read_text())

    def test_fmd_command_generates_fmd_identity_and_new_output_name(self):
        self.assertEqual(main(['train-fmd', '--project-root', str(self.root), '--steps', '750', '--no-eval']), 0)
        arguments = self.captured()['args']
        self.assertEqual(arguments[arguments.index('--model') + 1], 'FMD')
        self.assertEqual(Path(arguments[arguments.index('--output-dir') + 1]).name, 'v1713_model_fmd_step750')
        self.assertIn('train-fmd', COMMANDS)
        self.assertIn('resume-fmd', COMMANDS)
        self.assertNotIn('train-fm', COMMANDS)

    def test_fmd_refuses_legacy_fm_resume_and_restores_fmd_settings(self):
        self.args.name = 'my_distillation'
        output = self.ctx.model()
        output.mkdir(parents=True)
        (output / 'style_run.json').write_text(json.dumps(dict(batch_size=2, lr=2e-6,
            max_steps=500, gradient_accumulation=16, seed=42)))
        settings = dict(variant='FM', rank=32, alpha=64, loss='FM')
        (output / 'lora_settings.json').write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, 'Old FM runs'):
            experiments.resume(self.ctx, 'FMD', [])
        settings.update(variant='FMD', loss='pretrained_flow_distillation-v1')
        (output / 'lora_settings.json').write_text(json.dumps(settings))
        experiments.resume(self.ctx, 'FMD', [])
        arguments = self.captured()['args']
        self.assertEqual(arguments[arguments.index('--model') + 1], 'FMD')
        self.assertEqual(arguments[arguments.index('--lr') + 1], '2e-06')
        self.assertIn('--resume', arguments)

    def test_actual_child_uses_checkout_paths_environment_and_s_loss_settings(self):
        with patch.dict(os.environ, {"PROTOSS_ROOT": "/wrong/project", "VIRTUAL_ENV": "/wrong/venv",
                                     "UV_PROJECT_ENVIRONMENT": "/wrong/uv", "HF_HUB_ENABLE_HF_TRANSFER": "1"}):
            self.ctx = Recipe(self.args, self.config)
            experiments.train(self.ctx, "S", ["--no-eval"])
        record = self.captured()
        arguments = record["args"]
        self.assertEqual(record["root"], str(self.root))
        self.assertEqual(record["hf_transfer"], "0")
        self.assertIsNone(record["venv"])
        self.assertIsNone(record["uv_env"])
        for flag, value in (("--model", "S"), ("--max-steps", "2000"), ("--save-steps", "500"),
                            ("--workers", "1"), ("--s-fmd-weight", "1"), ("--s-sparc-weight", "0.001")):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        output = self.ctx.checkpoints / "v1712_model_s_step2000"
        self.assertEqual(arguments[arguments.index("--output-dir") + 1], str(output))
        self.assertFalse(output.exists())  # launch metadata must not make train.py reject a new run
        self.assertTrue((self.ctx.checkpoints / "launches/v1712_model_s_step2000.json").is_file())

    def test_resume_uses_saved_loss_and_lr_after_config_changes(self):
        experiments.train(self.ctx, "S", ["--no-eval"])
        self.config.TRAIN["S"].update(lr=1., s_sparc_weight=999, max_steps=7)
        experiments.resume(self.ctx, "S", [])
        arguments = self.captured()["args"]
        self.assertEqual(arguments[arguments.index("--lr") + 1], "1e-06")
        self.assertEqual(arguments[arguments.index("--s-sparc-weight") + 1], "0.001")
        self.assertIn("--resume", arguments)

    def test_step_override_also_changes_generated_output_name(self):
        experiments.train(self.ctx, "E", ["--max-steps", "1000"])
        arguments = self.captured()["args"]
        self.assertEqual(Path(arguments[arguments.index("--output-dir") + 1]).name, "v1712_model_e_step1000")

    def test_cli_accepts_name_steps_lr_and_loss_without_separator(self):
        result = main(["train-s", "--project-root", str(self.root), "--model-name", "from_cli",
                       "--steps", "750", "--save-steps", "100", "--lr", "3e-6",
                       "--s-fmd-weight", "1000", "--s-sparc-weight", "0.02", "--no-eval"])
        self.assertEqual(result, 0)
        arguments = self.captured()["args"]
        for flag, expected in (("--output-dir", str(self.ctx.checkpoints / "from_cli")),
                               ("--max-steps", "750"), ("--save-steps", "100"), ("--lr", "3e-6"),
                               ("--s-fmd-weight", "1000"), ("--s-sparc-weight", "0.02")):
            values = [arguments[i + 1] for i, token in enumerate(arguments) if token == flag]
            self.assertEqual(values[-1], expected)
        self.assertIn("--no-eval", arguments)

    def test_cli_steps_alias_updates_generated_name_and_resume(self):
        self.assertEqual(main(["train-e", "--project-root", str(self.root), "--step", "800"]), 0)
        arguments = self.captured()["args"]
        self.assertEqual(Path(arguments[arguments.index("--output-dir") + 1]).name,
                         "v1712_model_e_step800")
        self.assertEqual(main(["resume-e", "--project-root", str(self.root), "--max-steps", "1200"]), 0)
        arguments = self.captured()["args"]
        self.assertEqual([arguments[i + 1] for i, token in enumerate(arguments) if token == "--max-steps"][-1], "1200")
        self.assertIn("--resume", arguments)

    def test_cli_still_accepts_explicit_separator(self):
        self.assertEqual(main(["train-fmd", "--project-root", str(self.root),
                               "--", "--max-steps", "600", "--no-eval"]), 0)
        self.assertIn("--no-eval", self.captured()["args"])

    def test_cli_rejects_invalid_steps_before_writing_launch(self):
        with self.assertRaises(SystemExit) as caught:
            main(["train-s", "--project-root", str(self.root), "--steps", "0"])
        self.assertEqual(caught.exception.code, 2)
        self.assertFalse(self.ctx.checkpoints.exists())

    def test_local_evaluation_does_not_query_hf_login(self):
        self.config.HF_ACCOUNT = None
        (self.ctx.tools / "evaluate_robolab.py").write_text((self.ctx.tools / "train.py").read_text())
        experiments.evaluate(self.ctx, ["--no-hf-upload"])
        arguments = self.captured()["args"]
        self.assertNotIn("--hf-repo-id", arguments)
        self.assertIn("--no-hf-upload", arguments)

    def test_all_training_variants_and_standalone_evaluation_pass_the_same_ten_tasks(self):
        for variant in ('FMD', 'S', 'E'):
            experiments.train(self.ctx, variant, [])
            arguments = self.captured()['args']
            start = arguments.index('--eval-tasks') + 1
            end = arguments.index('--eval-num-envs')
            self.assertEqual(arguments[start:end], list(TEST_TASKS))
            self.assertEqual(arguments[end + 1], '1')
            self.assertEqual(arguments[arguments.index('--eval-num-runs') + 1], '10')
        (self.ctx.tools / 'evaluate_robolab.py').write_text((self.ctx.tools / 'train.py').read_text())
        experiments.evaluate(self.ctx, ['--no-hf-upload'])
        arguments = self.captured()['args']
        self.assertEqual(arguments[arguments.index('--eval-tasks') + 1:arguments.index('--eval-num-envs')], list(TEST_TASKS))
        self.assertEqual(arguments[arguments.index('--eval-num-runs') + 1], '10')

    def test_nonempty_output_is_preserved_without_launch_or_process(self):
        self.args.name = "existing"
        output = self.ctx.checkpoints / "existing"
        output.mkdir(parents=True)
        (output / "model.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            experiments.train(self.ctx, "S", [])
        self.assertEqual((output / "model.txt").read_text(), "preserve")
        self.assertFalse((self.root / "captured.json").exists())

    def test_failed_client_never_processes_or_uploads_metrics(self):
        (self.ctx.stack / "RoboLab").mkdir(parents=True)
        (self.ctx.tools / "setup_robolab_595.sh").write_text("#!/bin/bash\nexit 17\n")
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            experiments.client(self.ctx, [])
        self.assertEqual(caught.exception.returncode, 17)
        self.assertFalse((self.root / "captured.json").exists())

    def test_client_uses_one_name_for_output_and_metric_input(self):
        calls = []
        self.ctx.run = lambda command, **kwargs: calls.append(list(map(str, command))) or ""
        experiments.client(self.ctx, [])
        self.assertEqual(len(calls), 2)
        name = calls[0][calls[0].index("--output-folder-name") + 1]
        results = calls[1][calls[1].index("--results-dir") + 1]
        self.assertEqual(results, str(self.ctx.stack / "RoboLab/output" / name))
        self.assertEqual(calls[1][calls[1].index("--hf-repo-id") + 1], "test-owner/" + self.config.MODEL_NAME)
        self.assertEqual(calls[0][calls[0].index('--task') + 1:calls[0].index('--num-envs')], list(TEST_TASKS))
        self.assertEqual(calls[0][calls[0].index('--num-envs') + 1], '1')
        self.assertEqual(calls[0][calls[0].index('--num-runs') + 1], '10')

    def test_legacy_resume_recovers_original_weights_and_lr(self):
        self.args.name = "old_s"
        output = self.ctx.model()
        output.mkdir(parents=True)
        (output / "style_run.json").write_text(json.dumps(dict(batch_size=2, lr=2e-6,
            max_steps=1200, gradient_accumulation=16, seed=42)))
        (output / "lora_settings.json").write_text(json.dumps(dict(variant="S", rank=32, alpha=64,
            model_s=dict(fm_weight=25., sparc_weight=.4, dt=1/15, sparc_padlevel=4, sparc_cutoff_hz=10.,
                         sparc_amplitude_threshold=.05, gradient_diagnostics=False))))
        experiments.resume(self.ctx, "S", [])
        arguments = self.captured()["args"]
        self.assertEqual(arguments[arguments.index("--s-fm-weight") + 1], "25.0")
        self.assertEqual(arguments[arguments.index("--s-sparc-weight") + 1], "0.4")
        self.assertEqual(arguments[arguments.index("--lr") + 1], "2e-06")

    def test_groot_setup_keeps_fixed_revision_and_submodule_http_options(self):
        (self.ctx.groot / ".git").mkdir()
        calls = []
        def run(command, **kwargs):
            calls.append((list(map(str, command)), kwargs))
            return setup.GROOT_REVISION if command == ["git", "rev-parse", "HEAD"] else ""
        self.ctx.run = run
        setup.setup_groot(self.ctx, [])
        command, kwargs = next(item for item in calls if "submodule" in item[0])
        self.assertIn("http.version=HTTP/1.1", command)
        self.assertIn("protocol.version=0", command)
        self.assertEqual(kwargs["env"]["GIT_CONFIG_GLOBAL"], "/dev/null")
        self.assertTrue(any("nvidia/Cosmos-Reason2-2B" in cmd for cmd, _ in calls))

    def test_all_dry_runs_have_no_subprocess_or_file_mutations(self):
        self.args.dry_run = True
        self.config.HF_ACCOUNT = None
        # update-project only checks whether a checkout marker exists.
        (self.root / ".git").mkdir()
        before = {p.relative_to(self.root) for p in self.root.rglob("*")}
        with patch("src.recipe_runtime.subprocess.run", side_effect=AssertionError("executed during dry-run")):
            for command in COMMANDS.values():
                command(self.ctx, [])
        self.assertEqual(before, {p.relative_to(self.root) for p in self.root.rglob("*")})
        self.assertFalse(self.ctx.checkpoints.exists())


if __name__ == "__main__":
    unittest.main()
