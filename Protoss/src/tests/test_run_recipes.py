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
                                    name=None, port=None, result_name=None)
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

    def test_internal_module_runs_from_groot_without_wrapper_script(self):
        source = self.ctx.tools / 'src'
        source.mkdir()
        (source / '__init__.py').touch()
        (source / 'probe.py').write_text('''
import json, os, sys
print(json.dumps({'arguments': sys.argv[1:], 'cwd': os.getcwd(), 'project': os.environ['PROTOSS_ROOT']}))
''')
        result = json.loads(self.ctx.groot_run('probe', ['--value', '42'], capture=True))
        self.assertEqual(result, {'arguments': ['--value', '42'], 'cwd': str(self.ctx.groot),
                                  'project': str(self.root)})

    def test_removed_fmd_training_commands_are_unavailable(self):
        self.assertNotIn('train-fmd', COMMANDS)
        self.assertNotIn('resume-fmd', COMMANDS)
        self.assertEqual(set(self.config.TRAIN), {'S', 'E'})

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
                            ("--workers", "1"), ("--s-fmd-weight", "1"), ("--s-jerk-lambda", "0.001")):
            self.assertEqual(arguments[arguments.index(flag) + 1], value)
        output = self.ctx.checkpoints / "v1.7.13_model_s_step2000"
        self.assertEqual(arguments[arguments.index("--output-dir") + 1], str(output))
        self.assertFalse(output.exists())  # launch metadata must not make train.py reject a new run
        self.assertTrue((self.ctx.checkpoints / "launches/v1.7.13_model_s_step2000.json").is_file())

    def test_resume_uses_saved_loss_and_lr_after_config_changes(self):
        experiments.train(self.ctx, "S", ["--no-eval"])
        self.config.TRAIN["S"].update(lr=1., s_jerk_lambda=999, max_steps=7)
        experiments.resume(self.ctx, "S", [])
        arguments = self.captured()["args"]
        self.assertEqual(arguments[arguments.index("--lr") + 1], "1e-06")
        self.assertEqual(arguments[arguments.index("--s-jerk-lambda") + 1], "0.001")
        self.assertIn("--resume", arguments)

    def test_resume_replaces_old_task_selection_with_current_testset(self):
        import argparse
        from src.training_evaluation import add_evaluation_args
        self.config.EVALUATION['tasks'] = [*TEST_TASKS, 'AnimalsInBinTask']
        self.config.EVALUATION['num_envs'] = 1
        self.config.EVALUATION['num_runs'] = 10
        experiments.train(self.ctx, 'S', [])
        self.config.EVALUATION['tasks'] = list(TEST_TASKS)
        self.config.EVALUATION['num_envs'] = 2
        self.config.EVALUATION['num_runs'] = 5
        experiments.resume(self.ctx, 'S', [])
        parser = argparse.ArgumentParser()
        add_evaluation_args(parser)
        args, _ = parser.parse_known_args(self.captured()['args'])
        self.assertEqual(args.eval_tasks, list(TEST_TASKS))
        self.assertEqual((args.eval_num_envs, args.eval_num_runs), (2, 5))

    def test_step_override_also_changes_generated_output_name(self):
        experiments.train(self.ctx, "E", ["--max-steps", "1000"])
        arguments = self.captured()["args"]
        self.assertEqual(Path(arguments[arguments.index("--output-dir") + 1]).name, "v1.7.13_model_e_step1000")

    def test_cli_accepts_name_steps_lr_and_loss_without_separator(self):
        result = main(["train-s", "--project-root", str(self.root), "--model-name", "from_cli",
                       "--steps", "750", "--save-steps", "100", "--lr", "3e-6",
                       "--s-fmd-weight", "1000", "--s-jerk-lambda", "0.02", "--no-eval"])
        self.assertEqual(result, 0)
        arguments = self.captured()["args"]
        for flag, expected in (("--output-dir", str(self.ctx.checkpoints / "from_cli")),
                               ("--max-steps", "750"), ("--save-steps", "100"), ("--lr", "3e-6"),
                               ("--s-fmd-weight", "1000"), ("--s-jerk-lambda", "0.02")):
            values = [arguments[i + 1] for i, token in enumerate(arguments) if token == flag]
            self.assertEqual(values[-1], expected)
        self.assertIn("--no-eval", arguments)

    def test_cli_steps_alias_updates_generated_name_and_resume(self):
        self.assertEqual(main(["train-e", "--project-root", str(self.root), "--step", "800",
                               "--e-speedup", "1.4", "--e-fast-weight", "3", "--e-limit-weight", ".2"]), 0)
        arguments = self.captured()["args"]
        self.assertEqual(Path(arguments[arguments.index("--output-dir") + 1]).name,
                         "v1.7.13_model_e_step800")
        for flag, expected in (("--e-speedup", "1.4"), ("--e-fast-weight", "3"), ("--e-limit-weight", ".2")):
            self.assertEqual([arguments[i+1] for i, token in enumerate(arguments) if token == flag][-1], expected)
        self.assertEqual(main(["resume-e", "--project-root", str(self.root), "--max-steps", "1200"]), 0)
        arguments = self.captured()["args"]
        self.assertEqual([arguments[i + 1] for i, token in enumerate(arguments) if token == "--max-steps"][-1], "1200")
        self.assertIn("--resume", arguments)

    def test_cli_still_accepts_explicit_separator(self):
        self.assertEqual(main(["train-s", "--project-root", str(self.root),
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

    def test_all_training_variants_and_standalone_evaluation_pass_the_same_five_tasks(self):
        for variant in ('S', 'E'):
            experiments.train(self.ctx, variant, [])
            arguments = self.captured()['args']
            start = arguments.index('--eval-tasks') + 1
            end = arguments.index('--eval-num-envs')
            self.assertEqual(arguments[start:end], list(TEST_TASKS))
            self.assertEqual(arguments[end + 1], '2')
            self.assertEqual(arguments[arguments.index('--eval-num-runs') + 1], '5')
        (self.ctx.tools / 'evaluate_robolab.py').write_text((self.ctx.tools / 'train.py').read_text())
        experiments.evaluate(self.ctx, ['--no-hf-upload'])
        arguments = self.captured()['args']
        self.assertEqual(arguments[arguments.index('--eval-tasks') + 1:arguments.index('--eval-num-envs')], list(TEST_TASKS))
        self.assertEqual(arguments[arguments.index('--eval-num-envs') + 1], '2')
        self.assertEqual(arguments[arguments.index('--eval-num-runs') + 1], '5')

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
        self.assertIn('src.robolab_results', calls[1])
        self.assertEqual(calls[0][calls[0].index('--task') + 1:calls[0].index('--num-envs')], list(TEST_TASKS))
        self.assertEqual(calls[0][calls[0].index('--num-envs') + 1], '2')
        self.assertEqual(calls[0][calls[0].index('--num-runs') + 1], '5')

    def test_legacy_resume_recovers_original_weights_and_lr(self):
        self.args.name = "old_s"
        output = self.ctx.model()
        output.mkdir(parents=True)
        (output / "style_run.json").write_text(json.dumps(dict(batch_size=2, lr=2e-6,
            max_steps=1200, gradient_accumulation=16, seed=42)))
        (output / "lora_settings.json").write_text(json.dumps(dict(variant="S", rank=32, alpha=64,
            loss="pretrained_flow_preservation+normalized_joint_jerk-v1",
            model_s=dict(fm_weight=25., jerk_lambda=.4, jerk_reference=2., calibration_batches=10,
                         jerk_reference_floor=1., dt=1/15, gradient_diagnostics=False))))
        experiments.resume(self.ctx, "S", [])
        arguments = self.captured()["args"]
        self.assertEqual(arguments[arguments.index("--s-fm-weight") + 1], "25.0")
        self.assertEqual(arguments[arguments.index("--s-jerk-lambda") + 1], "0.4")
        self.assertEqual(arguments[arguments.index("--s-jerk-reference") + 1], "2.0")
        self.assertNotIn("--s-jerk-reference-floor", arguments)
        self.assertEqual(arguments[arguments.index("--lr") + 1], "2e-06")

        settings_path = output / 'lora_settings.json'
        settings = json.loads(settings_path.read_text())
        settings['loss'] = 'pretrained_flow_preservation+joint_sparc-v1'
        settings_path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, 'SPARC checkpoints'):
            experiments.resume(self.ctx, 'S', [])

    def test_model_e_resume_without_launch_restores_fast_settings_and_rejects_old_path_loss(self):
        self.args.name = 'fast_e'
        output = self.ctx.model()
        output.mkdir(parents=True)
        (output / 'style_run.json').write_text(json.dumps(dict(batch_size=2, lr=2e-6,
            max_steps=1200, gradient_accumulation=16, seed=42)))
        settings = dict(variant='E', rank=32, alpha=64,
            loss='pretrained_flow_preservation+fast_trajectory-v1',
            model_e=dict(fm_weight=25., fast_weight=.4, limit_weight=.2, speedup=1.5,
                         dt=1/15, velocity_limit=1.8, acceleration_limit=8.))
        path = output / 'lora_settings.json'
        path.write_text(json.dumps(settings))
        experiments.resume(self.ctx, 'E', [])
        arguments = self.captured()['args']
        for flag, value in (('--e-fmd-weight', '25.0'), ('--e-fast-weight', '0.4'),
                            ('--e-speedup', '1.5'), ('--e-limit-weight', '0.2')):
            self.assertEqual(arguments[arguments.index(flag)+1], value)
        settings['loss'] = 'pretrained_flow_preservation+fk_ee_path_length-v1'
        path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, 'start a new run'):
            experiments.resume(self.ctx, 'E', [])

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

    def hf_setup_fixture(self, logged_in=False, login_mode="ok"):
        """Run authentication checks in real children without network/tokens."""
        (self.ctx.groot / '.git').mkdir()
        if logged_in:
            (self.ctx.groot / 'logged_in').touch()
        (self.ctx.groot / 'huggingface_hub.py').write_text('''
from pathlib import Path
class HfApi:
    def whoami(self):
        with Path('auth_calls').open('a') as file: file.write('verify\\n')
        if not Path('logged_in').exists(): raise RuntimeError('Not logged in')
        return {'name': 'test-owner'}
''')
        stub = self.ctx.groot / 'uv_stub.py'
        stub.write_text('''
import os, sys
from pathlib import Path
command = sys.argv[1:]
assert command[:2] == ['run', '--no-sync']
command = command[2:]
if command[0] == 'python': os.execv(sys.executable, [sys.executable, *command[1:]])
assert command[0] == 'hf'
if command[1:] == ['auth', 'whoami']:
    print('test-owner' if Path('logged_in').exists() else 'Not logged in')
    sys.exit(0)  # Reproduce the old HF CLI's misleading success status.
with Path('auth_calls').open('a') as file: file.write(command[1] + '\\n')
if command[1:] == ['auth', 'login']:
    if LOGIN_MODE == 'failure': sys.exit(7)
    if LOGIN_MODE != 'unverified': Path('logged_in').touch()
elif command[1] == 'download':
    assert Path('logged_in').exists(), 'download attempted before login'
'''.replace('LOGIN_MODE', repr(login_mode)))
        original_run = self.ctx.run
        def run(command, **kwargs):
            if command[0] == 'uv' and command[1:3] == ['run', '--no-sync']:
                return original_run([sys.executable, stub, *command[1:]], **kwargs)
            return setup.GROOT_REVISION if command == ['git', 'rev-parse', 'HEAD'] else ''
        self.ctx.run = run

    def auth_calls(self):
        return (self.ctx.groot / 'auth_calls').read_text().splitlines()

    def test_setup_logs_in_when_old_cli_reports_not_logged_in_with_exit_zero(self):
        self.hf_setup_fixture()
        setup.hf(self.ctx, ['auth', 'whoami'])  # Returns successfully while logged out.
        setup.setup_groot(self.ctx, [])
        self.assertEqual(self.auth_calls(), ['verify', 'auth', 'verify', 'download', 'download'])

    def test_setup_reuses_valid_login_without_prompt(self):
        self.hf_setup_fixture(logged_in=True)
        setup.setup_groot(self.ctx, [])
        self.assertEqual(self.auth_calls(), ['verify', 'download', 'download'])

    def test_setup_does_not_download_if_login_fails_or_cannot_be_verified(self):
        self.hf_setup_fixture(login_mode='unverified')
        with self.assertRaises(subprocess.CalledProcessError):
            setup.setup_groot(self.ctx, [])
        self.assertEqual(self.auth_calls(), ['verify', 'auth', 'verify'])

    def test_failed_login_stops_setup_before_download(self):
        self.hf_setup_fixture(login_mode='failure')
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            setup.setup_groot(self.ctx, [])
        self.assertEqual(caught.exception.returncode, 7)
        self.assertEqual(self.auth_calls(), ['verify', 'auth'])

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
