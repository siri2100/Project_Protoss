from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.action_distribution import plot_distributions


class DistributionTests(unittest.TestCase):
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
