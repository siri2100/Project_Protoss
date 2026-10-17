from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.action_distribution import plot_distributions


class DistributionTests(unittest.TestCase):
    def test_training_uses_exported_run_and_training_data(self):
        from src.training_runtime import save_training_distribution_plot
        args = SimpleNamespace(groot_root=Path('/groot'), base_model=Path('/base'),
                               output_dir=Path('/checkpoint/run'), styles_dir=Path('/data'), seed=17)
        with patch('src.action_distribution.compare') as compare, \
             patch('src.gr00t_adapter_patch.ensure_adapter_inference') as ensure, \
             patch('torch.cuda.empty_cache'):
            save_training_distribution_plot(args)
        ensure.assert_called_once_with(args.groot_root)
        settings = compare.call_args.args[0]
        self.assertEqual(settings.model_path, args.output_dir)
        self.assertEqual(settings.output_dir, args.output_dir)
        self.assertEqual(settings.dataset_root, args.styles_dir)
        self.assertEqual((settings.split, settings.seed, settings.frames_per_trajectory), ('train', 17, 32))

    def test_plot_multidimensional_and_constant_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            before = {"arm": np.zeros((2, 1, 4, 3)), "gripper": np.ones((2, 1, 4, 1))}
            after = {"arm": np.full((2, 1, 4, 3), .1), "gripper": np.ones((2, 1, 4, 1))}
            path = plot_distributions(before, after, Path(directory))
            self.assertEqual(path.name, "distribution_plot.png")
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_rejects_incompatible_or_nonfinite_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = {"arm": np.zeros((2, 4, 3))}
            for invalid in ({}, {"arm": np.zeros((1, 4, 3))},
                            {"arm": np.full((2, 4, 3), np.nan)}):
                with self.assertRaises(ValueError):
                    plot_distributions(valid, invalid, directory)
            self.assertFalse((Path(directory) / "distribution_plot.png").exists())


if __name__ == "__main__":
    unittest.main()
