"""Keep recorder values compatible with tensor math and RoboLab's HDF5 schema."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.robolab_isaac60_compat import record_end_effector_state


class ProxyArray:
    def __init__(self, tensor):
        self.torch = tensor

    def __getitem__(self, key):
        raise AssertionError("Convert to a tensor before slicing")


class EndEffectorRecordingTests(unittest.TestCase):
    def check_recording(self, wrap):
        data = SimpleNamespace(
            root_pos_w=wrap(torch.tensor([[1., 2., 3.], [4., 5., 6.]])),
            root_quat_w=wrap(torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1)),
            body_pos_w=wrap(torch.arange(18.).reshape(2, 3, 3)),
            body_quat_w=wrap(torch.tensor([0., 0., 0., 1.]).repeat(2, 3, 1)),
            body_lin_vel_w=wrap(torch.arange(18.).reshape(2, 3, 3)),
            body_ang_vel_w=wrap(-torch.arange(18.).reshape(2, 3, 3)),
        )
        position = torch.tensor([[2., 3., 4.], [5., 6., 7.]])
        # Non-identity xyzw output makes a wrong recording convention detectable.
        orientation = torch.tensor([[0., 0., .6, .8], [0., .6, 0., .8]])

        def tensor_only_transform(root_pos, root_quat, ee_pos, ee_quat):
            for value in (root_pos, root_quat, ee_pos, ee_quat):
                self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(ee_pos.shape, (2, 3))
            self.assertEqual(ee_quat.shape, (2, 4))
            self.assertTrue(torch.equal(ee_pos, torch.tensor([[3., 4., 5.], [12., 13., 14.]])))
            return position, orientation

        module = SimpleNamespace(subtract_frame_transforms=tensor_only_transform)
        with patch.dict("sys.modules", {"isaaclab.utils.math": module}):
            recorded = record_end_effector_state(SimpleNamespace(data=data), 1)
        torch.testing.assert_close(recorded["position"], position)
        torch.testing.assert_close(recorded["orientation"], torch.tensor([[.8, 0., 0., .6], [.8, 0., .6, 0.]]))
        expected_velocity = torch.tensor([[3., 4., 5.], [12., 13., 14.]])
        torch.testing.assert_close(recorded["linear_velocity"], expected_velocity)
        torch.testing.assert_close(recorded["angular_velocity"], -expected_velocity)
        # Physics updates must not mutate values already handed to the recorder.
        linear = getattr(data.body_lin_vel_w, "torch", data.body_lin_vel_w)
        angular = getattr(data.body_ang_vel_w, "torch", data.body_ang_vel_w)
        linear.zero_()
        angular.zero_()
        torch.testing.assert_close(recorded["linear_velocity"], expected_velocity)
        torch.testing.assert_close(recorded["angular_velocity"], -expected_velocity)

    def test_proxy_arrays_are_converted_before_frame_math_and_indexing(self):
        self.check_recording(ProxyArray)

    def test_tensor_values_preserve_the_same_recording_schema(self):
        self.check_recording(lambda tensor: tensor)


if __name__ == "__main__":
    unittest.main()
