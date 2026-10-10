import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np


class BaseClient:
    def __init__(self):
        self._chunks, self._counters = {}, {}
    def _needs_refresh(self, env_id):
        return env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon
    def _set_chunk(self, env_id, chunk):
        self._chunks[env_id], self._counters[env_id] = chunk, 0
    def _next_action(self, env_id):
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action
    def reset(self, *, env_id=None):
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)


class VideoHistoryTests(unittest.TestCase):
    def test_history_chunk_steps_environment_isolation_and_reset(self):
        modules = {name: types.ModuleType(name) for name in
                   ('cv2', 'msgpack', 'msgpack_numpy', 'zmq', 'robolab',
                    'robolab.eval', 'robolab.eval.base_client')}
        modules['robolab.eval.base_client'].InferenceClient = BaseClient
        path = Path(__file__).parents[1] / 'robolab_gr00t_client.py'
        spec = importlib.util.spec_from_file_location('history_client_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        client = module.GR00TDroidJointposClient.__new__(module.GR00TDroidJointposClient)
        BaseClient.__init__(client)
        client.open_loop_horizon = 8
        client._video_history = {}
        requests = []
        def extract(step, *, env_id=0):
            image = np.full((180, 320, 3), step + env_id * 50, np.uint8)
            return {'external_image': image, 'wrist_image': image,
                    'eef_9d': np.zeros(9, np.float32),
                    'joint_position': np.zeros(7, np.float32),
                    'gripper_position': np.zeros(1, np.float32)}
        client._extract_observation = extract
        client._query_server = lambda request: requests.append(request)
        client._unpack_response = lambda response: np.zeros((8, 8), np.float32)
        client._build_visualization = lambda extracted: None
        for step in range(17):
            client.infer(step, 'task', env_id=0)
            client.infer(step, 'task', env_id=1)
        self.assertEqual(len(requests), 6)
        for index, expected in ((0, [0, 0]), (2, [0, 8]), (4, [1, 16]), (5, [51, 66])):
            video = requests[index]['video.exterior_image_1_left']
            self.assertEqual(video.shape, (1, 2, 180, 320, 3))
            self.assertEqual(video.dtype, np.uint8)
            self.assertEqual(video[0, :, 0, 0, 0].tolist(), expected)
            self.assertEqual(requests[index]['state.joint_position'].shape, (1, 1, 7))
        client.reset(env_id=0)
        self.assertIn(1, client._video_history)
        client.infer(30, 'new task', env_id=0)
        self.assertEqual(requests[-1]['video.wrist_image_left'][0, :, 0, 0, 0].tolist(), [30, 30])
        client.reset()
        self.assertFalse(client._video_history)
        self.assertFalse(client._chunks)
        # Avoid touching the real ZMQ transport in the destructor.
        client.client = types.SimpleNamespace(close=lambda: None)
