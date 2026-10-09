import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src import train_groot_models as training


class ModelTrainingTests(unittest.TestCase):
    def test_three_variants_reach_same_training_with_distinct_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "styles.json").write_text(json.dumps({"style_method": "original",
                "synthetic": False, "splits": {"train": [1], "test": [2]}}))
            settings = []
            for variant in ("1", "S", "E"):
                argv = ["train", "--model", variant, "--groot-root", str(root),
                    "--dataset-root", str(root), "--base-model", str(root / "base"),
                    "--output-dir", str(root / ("model_" + variant))]
                with patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)), patch.object(training, "train") as run:
                    training.main()
                    run.assert_called_once()
                    args = run.call_args.args[0]
                    self.assertEqual(args.variant, variant)
                    self.assertEqual(args.style, variant)
                    self.assertTrue(args.lora)
                    self.assertEqual(args.output_dir, root / ("model_" + variant))
                    settings.append((args.styles_dir, args.base_model, args.lora_rank,
                        args.lora_alpha, args.seed, args.lr, args.max_steps,
                        args.batch_size, args.gradient_accumulation))
            self.assertEqual(settings[0], settings[1])
            self.assertEqual(settings[1], settings[2])


if __name__ == "__main__":
    unittest.main()
