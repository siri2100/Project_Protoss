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


def load_client_module():
    modules = {name: types.ModuleType(name) for name in
               ('cv2', 'msgpack', 'msgpack_numpy', 'zmq', 'robolab',
                'robolab.eval', 'robolab.eval.base_client')}
    modules['robolab.eval.base_client'].InferenceClient = BaseClient
    path = Path(__file__).parents[1] / 'robolab_gr00t_client.py'
    spec = importlib.util.spec_from_file_location('history_client_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class VideoHistoryTests(unittest.TestCase):
    def test_nonfinite_observations_and_actions_are_not_sent_to_the_robot(self):
        module = load_client_module()
        with self.assertRaisesRegex(ValueError, "quaternion contains NaN/Inf"):
            module.compute_eef_9d(np.zeros(3), [1, np.nan, 0, 0])
        with self.assertRaisesRegex(ValueError, "position contains NaN/Inf"):
            module.compute_eef_9d([np.inf, 0, 0], [1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "joint_position.*NaN/Inf"):
            module._as_action_chunk([[0, np.nan]], name="joint_position")
        with self.assertRaisesRegex(ValueError, "gripper_position.*NaN/Inf"):
            module._as_action_chunk([np.nan], name="gripper_position")

    def test_instruction_mapping_is_sent_as_text_and_only_default_is_read(self):
        module = load_client_module()
        image = np.zeros((180, 320, 3), np.uint8)
        extracted = {'video_frames': [(image, image)],
                     'eef_9d': np.zeros(9, np.float32),
                     'joint_position': np.zeros(7, np.float32),
                     'gripper_position': np.zeros(1, np.float32)}
        reads = []
        class Fields(dict):
            def __getitem__(self, key):
                reads.append(key)
                return super().__getitem__(key)

        # Call the stateless packer directly; no simulator or socket is needed.
        fields = Fields(default='Pick up the banana and put it on the plate',
                        privileged_context='must not be read or sent')
        request = module.GR00TDroidJointposClient._pack_request(None, extracted, fields)
        self.assertEqual(reads, ['default'])
        self.assertEqual(request['annotation.language.language_instruction'], [fields['default']])
        self.assertIsInstance(request['annotation.language.language_instruction'][0], str)
        string_request = module.GR00TDroidJointposClient._pack_request(None, extracted, 'plain text')
        self.assertEqual(string_request['annotation.language.language_instruction'], ['plain text'])
        with self.assertRaisesRegex(ValueError, "'default'"):
            module.GR00TDroidJointposClient._pack_request(None, extracted, {'other': 'task'})
        with self.assertRaisesRegex(TypeError, 'string'):
            module.GR00TDroidJointposClient._pack_request(None, extracted, {'default': {'text': 'task'}})

    def test_history_chunk_steps_environment_isolation_and_reset(self):
        module = load_client_module()
        client = module.GR00TDroidJointposClient.__new__(module.GR00TDroidJointposClient)
        BaseClient.__init__(client)
        client.open_loop_horizon = 8
        client._video_history = {}
        client.video_delta_indices = (-15, 0)
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
        client.video_delta_indices = (0,)
        client.infer(40, {'default': 'single frame'}, env_id=0)
        self.assertEqual(requests[-1]['annotation.language.language_instruction'], ['single frame'])
        self.assertEqual(requests[-1]['video.wrist_image_left'].shape, (1, 1, 180, 320, 3))
        self.assertEqual(requests[-1]['video.wrist_image_left'][0, 0, 0, 0, 0], 40)
        # Check the native server's ModalityConfig envelope is understood.
        wire = {'video': {'__ModalityConfig__': True, 'as_json': {'delta_indices': [0]}}}
        transport = module.GR00TPolicyClient.__new__(module.GR00TPolicyClient)
        transport.api_token = None
        transport.socket = types.SimpleNamespace(send=lambda payload: None, recv=lambda: b'config')
        with patch.object(module._MsgSerializer, 'to_bytes', return_value=b'request'), patch.object(module._MsgSerializer, 'from_bytes', return_value=wire):
            self.assertEqual(transport.get_video_delta_indices(), (0,))
        # Avoid touching the real ZMQ transport in the destructor.
        client.client = types.SimpleNamespace(close=lambda: None)
