"""CPU tests with the real OpenPI codec/preprocessor and fake model outputs."""

import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from openpi_client import msgpack_numpy as pi_codec


spec = importlib.util.spec_from_file_location("protoss_v11", Path(__file__).with_name("main_v1.1.py"))
protoss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protoss)


def config():
    return {
        "video": {"delta_indices": [-15, 0], "modality_keys": ["exterior_image_1_left", "wrist_image_left"]},
        "state": {"delta_indices": [0], "modality_keys": ["eef_9d", "gripper_position", "joint_position"]},
        "language": {"delta_indices": [0], "modality_keys": ["instruction"]},
        "action": {"delta_indices": list(range(40)),
                   "modality_keys": ["eef_9d", "gripper_position", "joint_position"],
                   "action_configs": [{"state_key": key} for key in ["eef_9d", "gripper_position", "joint_position"]]},
    }


def observation():
    image = np.zeros((2, 2, 180, 320, 3), dtype=np.uint8)
    image[0, 1] = 200
    image[1, 1] = 100
    return {
        "video.exterior_image_1_left": image.copy(),
        "video.wrist_image_left": image.copy(),
        "state.joint_position": np.full((2, 1, 7), 0.1, np.float32),
        "state.gripper_position": np.full((2, 1, 1), 0.2, np.float32),
        "state.eef_9d": np.zeros((2, 1, 9), np.float32),
        "instruction": np.array(["pick banana", "place banana"]),
    }


def actions(value, batch=2):
    return {f"action.{name}": np.full((batch, 40, dim), value, np.float32)
            for name, dim in (("joint_position", 7), ("gripper_position", 1))}


class FakeN17:
    def __init__(self, cfg=None):
        self.cfg = cfg or config()
        self.calls = []

    def call(self, endpoint, data=None):
        self.calls.append((endpoint, data))
        if endpoint == "get_modality_config":
            return copy.deepcopy(self.cfg)
        if endpoint == "get_action":
            return actions(10), {}
        return {}


class FakePi:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        value = 2 if request["prompt"] == "pick banana" else 4
        return {"actions": np.full((15, 8), value, np.float32)}

    def reset(self):
        return {"stateless": True}


class FakeSocket:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.sent, self.timeouts = [], []
        self.closed = False

    def recv(self, timeout):
        self.timeouts.append(timeout)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def send(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True


class V11Tests(unittest.TestCase):
    def test_current_frame_letterbox_unbatched_inputs(self):
        obs = observation()
        requests = protoss.pi_observations(obs, config())
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["prompt"], "pick banana")
        self.assertEqual(requests[1]["prompt"], "place banana")
        image = requests[0]["observation/exterior_image_1_left"]
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image[112, 112], 200)
        np.testing.assert_array_equal(image[0, 0], 0)
        self.assertEqual(requests[0]["observation/joint_position"].shape, (7,))
        np.testing.assert_array_equal(obs["video.wrist_image_left"][0, 1], 200)

    def test_invalid_observations(self):
        obs = observation()
        obs["instruction"] = ["one prompt"]
        with self.assertRaises(ValueError):
            protoss.pi_observations(obs, config())
        obs = observation()
        obs["video.wrist_image_left"] = obs["video.wrist_image_left"].astype(float)
        with self.assertRaises(ValueError):
            protoss.pi_observations(obs, config())

    def test_pi_action_split(self):
        chunk = np.arange(120, dtype=np.float32).reshape(15, 8)
        result = protoss.pi_actions_to_dict([{"actions": chunk}], 8)
        np.testing.assert_array_equal(result["action.joint_position"][0], chunk[:8, :7])
        np.testing.assert_array_equal(result["action.gripper_position"][0], chunk[:8, 7:8])

    def test_bad_pi_chunks(self):
        for chunk in (np.zeros((1, 15, 8)), np.zeros((15, 7)),
                      np.zeros((7, 8)), np.full((15, 8), np.nan)):
            with self.assertRaises(ValueError):
                protoss.pi_actions_to_dict([{"actions": chunk}], 8)

    def test_blend_endpoints_and_weights(self):
        a17, api = actions(10), actions(2)
        for alpha, expected in ((0, 2), (0.25, 4), (1, 10)):
            result = protoss.blend_actions(a17, api, alpha, 8)
            for key in result:
                np.testing.assert_array_equal(result[key], expected)
                self.assertEqual(result[key].shape[1], 8)
                self.assertEqual(result[key].dtype, np.float32)
                self.assertFalse(np.shares_memory(result[key], a17[key]))

    def test_blend_rejects_invalid_data(self):
        for alpha in (float("nan"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                protoss.blend_actions(actions(1), actions(2), alpha)
        with self.assertRaises(ValueError):
            protoss.blend_actions(actions(1), actions(2, batch=1))

    def test_policy_batch_and_no_double_absolute_conversion(self):
        n17, pi = FakeN17(), FakePi()
        policy = protoss.BlendedPolicy(n17, pi, alpha=0.25)
        result, info = policy.get_action(observation())
        np.testing.assert_array_equal(result["action.joint_position"][0], 4)
        np.testing.assert_array_equal(result["action.joint_position"][1], 5.5)
        self.assertEqual(len(pi.requests), 2)
        self.assertEqual(info["models"][1], "pi05_droid_jointpos")
        advertised = policy.get_modality_config()["action"]["as_json"]
        self.assertEqual(advertised["modality_keys"], ["joint_position", "gripper_position"])
        self.assertEqual(advertised["delta_indices"], list(range(8)))
        self.assertEqual(policy.reset()["pi05"], {"stateless": True})
        with self.assertRaises(ValueError):
            policy.get_action(observation(), {"rtc_overlap_steps": 1})

    def test_both_servers_called_at_endpoints(self):
        for alpha in (0, 1):
            n17, pi = FakeN17(), FakePi()
            protoss.BlendedPolicy(n17, pi, alpha).get_action(observation())
            self.assertEqual(len(pi.requests), 2)
            self.assertTrue(any(name == "get_action" for name, _ in n17.calls))

    def test_horizon_and_time_offsets(self):
        with self.assertRaises(ValueError):
            protoss.BlendedPolicy(FakeN17(), FakePi(), horizon=16)
        cfg = config()
        cfg["action"]["delta_indices"] = list(range(1, 41))
        with self.assertRaises(ValueError):
            protoss.BlendedPolicy(FakeN17(cfg), FakePi())

    def test_websocket_real_codec_metadata_and_request(self):
        socket = FakeSocket([pi_codec.packb({"model": "test"}), pi_codec.packb({"actions": np.ones((15, 8))})])
        with patch("websockets.sync.client.connect", return_value=socket) as connect:
            client = protoss.OpenPiClient(timeout_ms=1500)
            self.assertIsNone(connect.call_args.kwargs["ping_interval"])
            response = client.infer({"prompt": "pick", "state": np.arange(7)})
            self.assertEqual(client.metadata, {"model": "test"})
            np.testing.assert_array_equal(pi_codec.unpackb(socket.sent[0])["state"], np.arange(7))
            self.assertEqual(response["actions"].shape, (15, 8))
            self.assertEqual(socket.timeouts, [1.5, 1.5])
            client.close()
            self.assertTrue(socket.closed)

    def test_websocket_timeout_reconnects_on_next_request(self):
        first = FakeSocket([pi_codec.packb({}), TimeoutError("test timeout")])
        second = FakeSocket([pi_codec.packb({}), pi_codec.packb({"actions": np.zeros((15, 8))})])
        with patch("websockets.sync.client.connect", side_effect=[first, second]) as connect:
            client = protoss.OpenPiClient()
            with self.assertRaises(TimeoutError):
                client.infer({"prompt": "first"})
            self.assertTrue(first.closed)
            self.assertEqual(connect.call_count, 1)  # No silent retry of failed action.
            client.infer({"prompt": "second"})
            self.assertEqual(connect.call_count, 2)
            client.close()

    def test_websocket_server_error(self):
        socket = FakeSocket([pi_codec.packb({}), "model failure"])
        with patch("websockets.sync.client.connect", return_value=socket):
            client = protoss.OpenPiClient()
            with self.assertRaisesRegex(RuntimeError, "model failure"):
                client.infer({"prompt": "test"})
            self.assertTrue(socket.closed)


if __name__ == "__main__":
    unittest.main()
