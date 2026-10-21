import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from src.inference_validation import decode_action_checked, rotation6d_to_matrix


class InferenceValidationTests(unittest.TestCase):
    def test_valid_prediction_keeps_decoder_arguments_and_output(self):
        processor = Mock()
        action = np.zeros((1, 16, 17), dtype=np.float32)
        states = {"joint_position": np.zeros((1, 1, 7))}
        result = decode_action_checked(processor, action, "droid", states)
        self.assertIs(result, processor.decode_action.return_value)
        args = processor.decode_action.call_args.args
        self.assertIs(args[0], action)
        self.assertIs(args[2], states)

    def test_nonfinite_prediction_reports_location_before_decoding(self):
        processor = Mock()
        action = np.zeros((1, 16, 17), dtype=np.float32)
        action[0, 3, 4] = np.nan
        with self.assertRaisesRegex(ValueError, r"action_pred.*\(0, 3, 4\)"):
            decode_action_checked(processor, action, "droid", {})
        processor.decode_action.assert_not_called()

    def test_nonfinite_reference_is_distinguished_from_model_failure(self):
        processor = Mock()
        with self.assertRaisesRegex(ValueError, "Reference state 'eef_9d'"):
            decode_action_checked(processor, np.zeros((1, 16, 17)), "droid", {"eef_9d": np.array([np.inf])})
        processor.decode_action.assert_not_called()

    def test_6d_rotation_matches_known_rotations_even_at_extreme_scales(self):
        for rotation in Rotation.random(32, random_state=42).as_matrix():
            for scales in ([1, 1], [1e300, 1e-300]):
                value = (rotation[:2] * np.array(scales)[:, None]).reshape(6)
                result = rotation6d_to_matrix(value)
                np.testing.assert_allclose(result, rotation, atol=1e-12)
                np.testing.assert_allclose(result @ result.T, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(result), 1.)

    def test_invalid_rotations_fail_explicitly_without_identity_fallback(self):
        for value, reason in (([np.nan, 0, 0, 0, 1, 0], "NaN/Inf"),
                              ([0, 0, 0, 0, 1, 0], "zero rotation row"),
                              ([1, 0, 0, 2, 0, 0], "parallel rotation rows")):
            with self.assertRaisesRegex(ValueError, reason):
                rotation6d_to_matrix(value)


if __name__ == "__main__":
    unittest.main()
