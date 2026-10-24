import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import train as training
from src.robolab_testset import TEST_TASKS


class ModelTrainingTests(unittest.TestCase):
    def test_two_variants_parse_same_settings_with_distinct_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "styles.json").write_text(json.dumps({"style_method": "original",
                "synthetic": False, "splits": {"train": [1]}}))
            settings = []
            for variant in ("S", "E"):
                argv = ["train", "--model", variant, "--groot-root", str(root),
                    "--dataset-root", str(root), "--base-model", str(root / "base"),
                    "--output-dir", str(root / ("model_" + variant))]
                with patch.object(sys, "argv", argv), patch.object(sys, "path", list(sys.path)):
                    args = training.parse_args()
                    self.assertEqual(args.variant, variant)
                    self.assertEqual(args.style, variant)
                    self.assertEqual(args.output_dir, root / ("model_" + variant))
                    self.assertEqual(args.s_fm_weight, 1.0)
                    self.assertFalse(hasattr(args, "s_jerk_weight"))
                    self.assertEqual(args.s_sparc_weight, .01)
                    self.assertEqual(args.e_fm_weight, 1.)
                    self.assertEqual(args.e_fast_weight, 1.)
                    self.assertEqual(args.e_limit_weight, .01)
                    self.assertEqual(args.e_speedup, 1.2)
                    self.assertFalse(args.no_eval)
                    self.assertEqual(args.eval_tasks, list(TEST_TASKS))
                    self.assertEqual((args.eval_num_envs, args.eval_num_runs, args.eval_horizon), (2, 5, 8))
                    settings.append((args.styles_dir, args.base_model, args.lora_rank,
                        args.lora_alpha, args.seed, args.lr, args.max_steps,
                        args.batch_size, args.gradient_accumulation))
            self.assertEqual(settings[0], settings[1])

            with patch.object(sys, "argv", [*argv, "--model", "S", "--s-sparc-weight", "0.25", "--s-fm-weight", "1000"]), \
                 patch.object(sys, "path", list(sys.path)):
                args = training.parse_args()
                from src.model_s_loss import config_from_args
                self.assertEqual(config_from_args(args).sparc_weight, .25)
                self.assertEqual(config_from_args(args).fm_weight, 1000)
            with patch.object(sys, "argv", [*argv, "--e-fmd-weight", "2000", "--e-fast-weight", ".2", "--e-limit-weight", ".03", "--e-speedup", "1.5", "--e-velocity-limit", "1.8", "--e-acceleration-limit", "8"]), \
                 patch.object(sys, "path", list(sys.path)):
                args = training.parse_args()
                from src.model_e_loss import config_from_args
                self.assertEqual(config_from_args(args).fm_weight, 2000)
                self.assertEqual(config_from_args(args).fast_weight, .2)
                self.assertEqual(config_from_args(args).limit_weight, .03)
                self.assertEqual(config_from_args(args).speedup, 1.5)
                self.assertEqual(config_from_args(args).velocity_limit, 1.8)
                self.assertEqual(config_from_args(args).acceleration_limit, 8.)
            with patch.object(sys, "argv", [*argv, "--no-eval", "--eval-num-runs", "1"]), \
                 patch.object(sys, "path", list(sys.path)):
                args = training.parse_args()
                self.assertTrue(args.no_eval)
                self.assertEqual(args.eval_num_runs, 1)

            with patch.object(sys, 'argv', [*argv, '--model', 'FMD']), self.assertRaises(SystemExit) as caught:
                training.parse_args()
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
