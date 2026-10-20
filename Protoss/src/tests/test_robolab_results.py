import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from src.robolab_results import process_results


class RoboLabResultsTests(unittest.TestCase):
    def write_results(self, root, records):
        (root / "episode_results.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n")

    def test_all_tasks_numbering_finite_values_and_episode_weighted_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_results(root, [
                {"task_name": "B", "success": True, "score": 1, "metrics": {"speed": 2, "flag": True}},
                {"task_name": "A", "success": False, "score": 0, "metrics": {"speed": 8}},
                {"task_name": "B", "success": False, "metrics": {"speed": 4, "bad": float("nan")}},
            ])
            (root / "run.hdf5").write_text("large trajectory")
            (root / "adapter_model.safetensors").write_text("weights")
            output = root / "metrics"
            output.mkdir()
            (output / "RoboLab99.jsonl").write_text("stale")
            api = Mock()
            def check_upload(**kwargs):
                artifacts = Path(kwargs["folder_path"])
                self.assertEqual({p.name for p in artifacts.iterdir()}, {
                    "episode_results.jsonl", "RoboLab09.jsonl", "RoboLab10.jsonl",
                    "metrics_summary.json", "metrics_summary.txt"})
                self.assertEqual(len((artifacts / "RoboLab09.jsonl").read_text().splitlines()), 2)
                self.assertEqual(kwargs["path_in_repo"], "evaluations/test-run")
                return "commit"
            api.upload_folder.side_effect = check_upload
            summary, commit = process_results(root, "account/model", 9, "test-run", api)
            self.assertEqual(commit, "commit")
            self.assertEqual([task["task_name"] for task in summary["tasks"]], ["B", "A"])
            self.assertEqual(summary["tasks"][0]["metric_averages"], {"speed": 3})
            self.assertAlmostEqual(summary["overall"]["metric_averages"]["speed"], 14 / 3)
            self.assertAlmostEqual(summary["overall"]["success_rate"], 1 / 3)
            self.assertEqual(json.loads((output / "metrics_summary.json").read_text()), summary)
            api.repo_info.assert_called_once_with(repo_id="account/model", repo_type="model")

    def test_missing_metrics_prevents_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_results(root, [{"task_name": "A", "metrics": {"speed": None}}])
            api = Mock()
            with self.assertRaisesRegex(ValueError, "No finite"):
                process_results(root, "account/model", api=api)
            api.upload_folder.assert_not_called()

    def test_invalid_input_prevents_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episode_results.jsonl").write_text('{"task_name":')
            api = Mock()
            with self.assertRaisesRegex(ValueError, "Invalid JSON"):
                process_results(root, "account/model", api=api)
            api.upload_folder.assert_not_called()

    def test_failed_upload_preserves_local_metrics_and_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_results(root, [{"env_name": "A", "metrics": {"speed": 2}}])
            api = Mock()
            api.upload_folder.side_effect = RuntimeError("network error")
            with self.assertRaisesRegex(RuntimeError, "network error"):
                process_results(root, "account/model", api=api)
            self.assertTrue((root / "metrics/metrics_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
