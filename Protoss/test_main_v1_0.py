"""CPU checks for blending and model protocol adaptation; no weights needed."""

import copy
import importlib.util
from pathlib import Path
import unittest

import msgpack
import msgpack_numpy as mnp
import numpy as np


spec = importlib.util.spec_from_file_location("protoss_v10", Path(__file__).with_name("main_v1.0.py"))
protoss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protoss)


def config(n17=True):
    names = ["eef_9d", "gripper_position", "joint_position"] if n17 else ["joint_position", "gripper_position"]
    return {
        "video": {"delta_indices": [-15, 0] if n17 else [0], "modality_keys": ["wrist_image_left"]},
        "state": {"delta_indices": [0], "modality_keys": names},
        "language": {"delta_indices": [0], "modality_keys": ["instruction"]},
        "action": {"delta_indices": list(range(40 if n17 else 50)), "modality_keys": names,
                   "action_configs": [{"state_key": name} for name in names]},
    }


def actions(value, horizon=10):
    return {f"action.{key}": np.full((2, horizon, dim), value, dtype=np.float32)
            for key, dim in protoss.ACTION_DIMS.items()}


class FakeClient:
    def __init__(self, cfg, value):
        self.cfg, self.value = cfg, value
        self.requests = []

    def call(self, endpoint, data=None):
        self.requests.append((endpoint, data))
        if endpoint == "get_modality_config":
            return copy.deepcopy(self.cfg)
        if endpoint == "get_action":
            return actions(self.value), {}
        if endpoint == "reset":
            return {}
        raise AssertionError(endpoint)


class BlendingTests(unittest.TestCase):
    def test_weight_and_horizon(self):
        result = protoss.blend_actions(actions(10, 40), actions(2, 50), alpha=0.25, horizon=8)
        for key, arr in result.items():
            np.testing.assert_array_equal(arr, np.full((2, 8, protoss.ACTION_DIMS[key[7:]]), 4, np.float32))
            self.assertEqual(arr.dtype, np.float32)

    def test_endpoints_and_no_mutation(self):
        a17, a16 = actions(3), actions(-2)
        for alpha, expected in ((1, a17), (0, a16)):
            result = protoss.blend_actions(a17, a16, alpha, 8)
            for key in result:
                np.testing.assert_array_equal(result[key], expected[key][:, :8])
                self.assertFalse(np.shares_memory(result[key], expected[key]))
        np.testing.assert_array_equal(a17["action.joint_position"], 3)

    def test_invalid_parameters(self):
        for alpha in (-1, 2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                protoss.blend_actions(actions(1), actions(2), alpha)
        with self.assertRaises(ValueError):
            protoss.blend_actions(actions(1), actions(2), horizon=0)

    def test_invalid_model_outputs(self):
        for invalid in (np.zeros((2, 8, 6)), np.zeros((1, 8, 7)),
                        np.zeros((2, 7, 7)), np.full((2, 8, 7), np.nan)):
            a16 = actions(2)
            a16["action.joint_position"] = invalid
            with self.assertRaises(ValueError):
                protoss.blend_actions(actions(1), a16)

    def test_wire_formats(self):
        original = {"video": np.arange(24, dtype=np.uint8).reshape(2, 2, 2, 3), "actions": actions(2)}
        for legacy in (False, True):
            recovered = protoss.decode(protoss.encode(original, legacy=legacy))
            np.testing.assert_array_equal(recovered["video"], original["video"])
            np.testing.assert_array_equal(recovered["actions"]["action.joint_position"], original["actions"]["action.joint_position"])
        # Verify N1.6 can decode the emitted legacy array without msgpack_numpy.
        legacy_envelope = msgpack.unpackb(protoss.encode(original["video"], legacy=True), raw=False)
        self.assertTrue(legacy_envelope["__ndarray_class__"])
        # Verify N1.7 can decode the emitted modern array using msgpack_numpy.
        np.testing.assert_array_equal(mnp.unpackb(protoss.encode(original["video"])), original["video"])

    def test_modality_formats_and_object_rejection(self):
        for marker in ("__ModalityConfig__", "__ModalityConfig_class__"):
            self.assertEqual(protoss.decode(protoss.encode({marker: True, "as_json": config()})), config())
        for legacy in (False, True):
            with self.assertRaises(TypeError):
                protoss.encode(np.array([{}], dtype=object), legacy=legacy)
        forged = msgpack.packb({b"nd": True, b"kind": b"O", b"data": b"invalid"})
        with self.assertRaises(ValueError):
            protoss.decode(forged)

    def test_end_to_end_policy_with_fake_models(self):
        c17, c16 = FakeClient(config(), 10), FakeClient(config(False), 2)
        policy = protoss.BlendedPolicy(c17, c16, alpha=0.25, horizon=8)
        video = np.zeros((2, 2, 2, 2, 3), np.uint8)
        video[:, 1] = 255
        observation = {"video.wrist_image_left": video,
                       "state.eef_9d": np.zeros((2, 1, 9), np.float32),
                       "state.joint_position": np.zeros((2, 1, 7), np.float32),
                       "state.gripper_position": np.zeros((2, 1, 1), np.float32),
                       "instruction": np.array(["pick", "place"])}
        result, info = policy.get_action(observation)
        np.testing.assert_array_equal(result["action.joint_position"], 4)
        obs16 = c16.requests[-1][1]["observation"]
        np.testing.assert_array_equal(obs16["video.wrist_image_left"], video[:, 1:2])
        self.assertNotIn("state.eef_9d", obs16)
        self.assertEqual(obs16["instruction"], ["pick", "place"])
        self.assertEqual(info["alpha"], 0.25)
        advertised = policy.get_modality_config()["action"]["as_json"]
        self.assertEqual(advertised["modality_keys"], ["joint_position", "gripper_position"])
        self.assertEqual(advertised["action_configs"], [{"state_key": "joint_position"}, {"state_key": "gripper_position"}])
        self.assertEqual(advertised["delta_indices"], list(range(8)))
        self.assertEqual(policy.reset(), {"n17": {}, "n16": {}})
        with self.assertRaises(ValueError):
            policy.get_action(observation, {"rtc_overlap_steps": 1})

    def test_missing_history_is_not_duplicated(self):
        observation = {"video.wrist_image_left": np.zeros((2, 1, 2, 2, 3), np.uint8)}
        with self.assertRaises(ValueError):
            protoss.adapt_observation(observation, config(), config())

    def test_reject_different_action_timestamps(self):
        cfg16 = config(False)
        cfg16["action"]["delta_indices"] = list(range(1, 51))
        with self.assertRaises(ValueError):
            protoss.BlendedPolicy(FakeClient(config(), 1), FakeClient(cfg16, 2))


if __name__ == "__main__":
    unittest.main()
