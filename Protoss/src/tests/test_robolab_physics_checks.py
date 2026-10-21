from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from src.robolab_physics_checks import checked_step


class PhysicsChecksTests(unittest.TestCase):
    def make_env(self):
        data = SimpleNamespace(joint_names=["joint1", "joint2", "joint3"],
                               joint_pos=torch.zeros(2, 3), joint_vel=torch.zeros(2, 3),
                               root_quat_w=torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1))
        return SimpleNamespace(scene={"robot": SimpleNamespace(data=data)},
                               cfg=SimpleNamespace(sim=SimpleNamespace(dt=1/120), seed=0))

    def test_valid_actions_and_return_value_are_unchanged(self):
        env = self.make_env()
        action = torch.ones(2, 4)
        advance = Mock(return_value=object())
        self.assertIs(checked_step(env, action, advance), advance.return_value)
        self.assertIs(advance.call_args.args[0], action)
        self.assertEqual(env._protoss_physics_history[0]["action"].tolist(), action.tolist())

    def test_first_invalid_state_saves_finite_command_history_and_joint_name(self):
        env = self.make_env()
        action = torch.ones(2, 4)
        checked_step(env, action, lambda value: None)

        def advance(value):
            env.scene["robot"].data.joint_pos[0, 1] = float("nan")

        with tempfile.TemporaryDirectory() as directory:
            constants = SimpleNamespace(get_output_dir=lambda: directory)
            with patch.dict("sys.modules", {"robolab.constants": constants}):
                with self.assertRaisesRegex(RuntimeError, r"after step: step=2.*joint=joint2"):
                    checked_step(env, action, advance)
            files = list(Path(directory).glob("physics_failure-*.npz"))
            self.assertEqual(len(files), 1)
            with np.load(files[0], allow_pickle=False) as saved:
                self.assertTrue(np.isfinite(saved["history_joint_position"]).all())
                self.assertEqual(saved["history_action"].shape, (2, 2, 4))
                self.assertTrue(np.isnan(saved["fault_joint_position"][0, 1]))
                self.assertEqual(saved["step"].item(), 2)

    def test_nonfinite_action_never_reaches_simulation(self):
        env = self.make_env()
        action = torch.zeros(2, 4)
        action[1, 0] = float("inf")
        advance = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"robolab.constants": SimpleNamespace(get_output_dir=lambda: directory)}):
                with self.assertRaisesRegex(RuntimeError, "before step.*field=action"):
                    checked_step(env, action, advance)
        advance.assert_not_called()

    def test_unrelated_step_error_is_preserved(self):
        env = self.make_env()
        error = ValueError("recorder bug")
        with self.assertRaises(ValueError) as caught:
            checked_step(env, torch.zeros(2, 4), Mock(side_effect=error))
        self.assertIs(caught.exception, error)

    def test_corruption_during_recorder_error_is_captured(self):
        env = self.make_env()

        def advance(value):
            env.scene["robot"].data.joint_vel[1, 2] = float("inf")
            raise ValueError("recorder saw corrupt state")

        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", {"robolab.constants": SimpleNamespace(get_output_dir=lambda: directory)}):
                with self.assertRaisesRegex(RuntimeError, "during step.*joint=joint3") as caught:
                    checked_step(env, torch.zeros(2, 4), advance)
            self.assertIsInstance(caught.exception.__context__, ValueError)
            self.assertEqual(len(list(Path(directory).glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
