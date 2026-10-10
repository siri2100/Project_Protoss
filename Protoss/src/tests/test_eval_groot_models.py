import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src import eval_groot_models as evaluation


class EvaluationTests(unittest.TestCase):
    def test_same_test_ids_seeds_and_horizon_for_all_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "styles.json").write_text(json.dumps({"style_method": "original", "synthetic": False,
                "splits": {"train": [{"source_episode": 0}], "test": [{"source_episode": 1}, {"source_episode": 2}]}}))
            commands = []
            for model in ("baseline", "FM", "S", "E"):
                argv = ["eval", "--model", model, "--groot-root", str(root), "--dataset-root", str(root),
                        "--model-path", str(root / ("model_" + model)), "--output-dir", str(root / ("eval_" + model))]
                with patch.object(sys, "argv", argv), patch.object(evaluation.subprocess, "run") as run:
                    evaluation.main()
                    calls = [call.args[0] for call in run.call_args_list]
                    self.assertEqual(len(calls), 3)
                    self.assertEqual([command[3] for command in calls], ["42", "43", "44"])
                    for command in calls:
                        self.assertEqual(command[command.index("--execution-horizon") + 1], "8")
                        self.assertEqual(command[command.index("--denoising-steps") + 1], "4")
                        start = command.index("--traj-ids") + 1
                        self.assertEqual(command[start:start + 2], ["0", "1"])
                    self.assertEqual(json.loads((root / ("eval_" + model) / "evaluation.json").read_text())["model"], model)
                    commands.append(json.loads((root / ("eval_" + model) / "evaluation.json").read_text()))
            self.assertEqual(len({c["dataset_manifest_sha256"] for c in commands}), 1)


if __name__ == "__main__":
    unittest.main()
