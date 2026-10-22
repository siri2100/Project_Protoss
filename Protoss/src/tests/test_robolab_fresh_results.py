"""Reproduce completed-result skips and ensure reruns use only fresh episodes."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from src import robolab_fresh_results as fresh
from src.robolab_results import process_results


class FreshResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output_root = self.root / "output"
        self.name = "same_model_step500"

    def modules(self):
        package = ModuleType("robolab")
        constants = ModuleType("robolab.constants")
        constants.PACKAGE_DIR = str(self.root)
        return {"robolab": package, "robolab.constants": constants}

    def test_existing_jsonl_hdf5_and_metrics_are_preserved_outside_fresh_run(self):
        previous = self.output_root / self.name
        (previous / "Task").mkdir(parents=True)
        (previous / "episode_results.jsonl").write_text("old episodes")
        (previous / "Task/run_0.hdf5").write_bytes(b"old trajectory")
        (previous / "metrics").mkdir()
        (previous / "metrics/metrics_summary.json").write_text("old score")
        with fresh.fresh_output_directory(self.output_root, self.name) as output:
            self.assertEqual(output, previous)
            self.assertFalse((output / "episode_results.jsonl").exists())
            self.assertFalse((output / "Task/run_0.hdf5").exists())
            self.assertFalse((output / "metrics").exists())
            manifest = json.loads((output / "evaluation_run.json").read_text())
            archive = Path(manifest["previous_output"])
            self.assertEqual((archive / "episode_results.jsonl").read_text(), "old episodes")
            self.assertEqual((archive / "Task/run_0.hdf5").read_bytes(), b"old trajectory")
            self.assertEqual((archive / "metrics/metrics_summary.json").read_text(), "old score")

    def test_same_name_evaluates_again_and_metrics_use_only_new_results(self):
        calls = []
        def evaluate(args, **kwargs):
            folder = self.output_root / args.output_folder_name
            # Mirrors RoboLab: loading a previous JSONL would skip saved episodes.
            source = folder / "episode_results.jsonl"
            saved = [json.loads(line) for line in source.read_text().splitlines()] if source.exists() else []
            if saved:
                return "skipped"
            calls.append(len(calls) + 1)
            source.write_text(json.dumps({"task_name": "Task", "success": True,
                "score": calls[-1], "metrics": {"path_length": calls[-1] * 10}}) + "\n")
            return "evaluated"
        wrapped = fresh.fresh_evaluation(evaluate)
        args = SimpleNamespace(output_folder_name=self.name)
        with patch.dict(sys.modules, self.modules()):
            self.assertEqual(wrapped(args, policy="gr00t"), "evaluated")
            process_results(self.output_root / self.name)
            self.assertEqual(wrapped(args, policy="gr00t"), "evaluated")
            summary, _ = process_results(self.output_root / self.name)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(summary["overall"]["records"], 1)
        self.assertEqual(summary["overall"]["average_score"], 2)
        self.assertEqual(summary["overall"]["metric_averages"], {"path_length": 20})

    def test_failed_run_is_archived_before_retry(self):
        with self.assertRaisesRegex(RuntimeError, "simulation error"):
            with fresh.fresh_output_directory(self.output_root, self.name) as folder:
                (folder / "episode_results.jsonl").write_text("partial")
                raise RuntimeError("simulation error")
        with fresh.fresh_output_directory(self.output_root, self.name) as folder:
            self.assertFalse((folder / "episode_results.jsonl").exists())
            archive = Path(json.loads((folder / "evaluation_run.json").read_text())["previous_output"])
            self.assertEqual((archive / "episode_results.jsonl").read_text(), "partial")

    def child(self, body):
        tools = Path(__file__).resolve().parents[2]
        source = f"import sys\nsys.path.insert(0, {str(tools)!r})\nfrom src.robolab_fresh_results import fresh_output_directory\n" + body
        return subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)

    def test_concurrent_same_name_does_not_archive_active_results(self):
        with fresh.fresh_output_directory(self.output_root, self.name) as folder:
            (folder / "active").write_text("in progress")
            result = self.child(f"with fresh_output_directory({str(self.output_root)!r}, {self.name!r}):\n    pass\n")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Evaluation already running", result.stderr)
            self.assertEqual((folder / "active").read_text(), "in progress")
            self.assertFalse((self.output_root / "_previous").exists())

    def test_native_exit_releases_lock_and_next_run_archives_partial_output(self):
        result = self.child(f"import os\nwith fresh_output_directory({str(self.output_root)!r}, {self.name!r}) as folder:\n    (folder / 'partial.hdf5').write_bytes(b'partial')\n    os._exit(0)\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        with fresh.fresh_output_directory(self.output_root, self.name) as folder:
            self.assertFalse((folder / "partial.hdf5").exists())
            archive = Path(json.loads((folder / "evaluation_run.json").read_text())["previous_output"])
            self.assertEqual((archive / "partial.hdf5").read_bytes(), b"partial")

    def test_entrypoint_patch_and_completion_patch_apply_together(self):
        tools = Path(__file__).resolve().parents[2]
        target = self.root / "policies/gr00t/run.py"
        target.parent.mkdir(parents=True)
        target.write_text('''from types import SimpleNamespace
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = SimpleNamespace(output_folder_name="same_model_step500")
POLICY = "gr00t"
make_client = None
simulation_app = SimpleNamespace(close=lambda: None)
def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()
''')
        patches = [tools / "patches" / name for name in ("robolab_eval_completion.patch", "robolab_fresh_results.patch")]
        for source in patches:
            subprocess.run(["git", "apply", str(source)], cwd=self.root, capture_output=True, check=True)
        for source in patches:
            subprocess.run(["git", "apply", "--reverse", "--check", str(source)], cwd=self.root, capture_output=True, check=True)
        modules = self.modules()
        modules["robolab.eval"] = ModuleType("robolab.eval")
        runner = ModuleType("robolab.eval.runner")
        runner.add_common_eval_args = lambda parser: None
        calls = []
        def evaluate(args, **kwargs):
            folder = self.output_root / args.output_folder_name
            self.assertFalse((folder / "episode_results.jsonl").exists())
            (folder / "episode_results.jsonl").write_text("new result")
            calls.append(1)
        runner.run_evaluation = evaluate
        modules["robolab.eval.runner"] = runner
        modules["robolab.protoss_fresh_results"] = fresh
        with patch.dict(sys.modules, modules):
            namespace = runpy.run_path(str(target), init_globals={"parser": None,
                "AppLauncher": SimpleNamespace(add_app_launcher_args=lambda parser: None)})
            namespace["main"]()
            namespace["main"]()
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
