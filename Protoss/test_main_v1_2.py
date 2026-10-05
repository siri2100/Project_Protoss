"""CPU integration tests; no pretrained server or dataset download."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('v12', Path(__file__).with_name('main_v1.2.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class RefinerTests(unittest.TestCase):
    def sample(self, directory, episode='train', mask=None):
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(1)
        blend = rng.normal(size=(2, 2, 8)).astype(np.float32)
        np.savez(directory / 'sample.npz', blend=blend, target=blend + .1,
                 state=blend[:, 0], mask=np.ones((2, 2)) if mask is None else mask,
                 episode_id=np.array([episode] * 2), alpha=.5, horizon=2)

    def test_loss_masks_padding_and_backpropagates(self):
        model = v.ResidualFlow(2, 16)
        blend, state = torch.zeros(1, 2, 8), torch.zeros(1, 8)
        target, mask = torch.ones_like(blend), torch.tensor([[1., 0.]])
        torch.manual_seed(3)
        first = model.loss(blend, state, target, mask)
        target[:, 1] = 10000
        torch.manual_seed(3)
        second = model.loss(blend, state, target, mask)
        torch.testing.assert_close(first, second)
        second.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_training_checkpoint_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.sample(root / 'train')
            self.sample(root / 'val', 'val')
            args = argparse.Namespace(train_cache=root/'train', val_cache=root/'val',
                alpha=.5, horizon=2, hidden=16, device='cpu', seed=42,
                lr=.001, batch_size=2, epochs=2, checkpoint=root/'model.pt')
            v.train_refiner(args)
            model = v.load_refiner(args.checkpoint, .5, 2, 'cpu', 3)
            actions = {'action.joint_position': np.zeros((2, 2, 7), np.float32),
                       'action.gripper_position': np.zeros((2, 2, 1), np.float32)}
            result = model.refine(actions, np.zeros((2, 8), np.float32))
            self.assertEqual(v.pack_actions(result).shape, (2, 2, 8))
            self.assertTrue(np.isfinite(v.pack_actions(result)).all())
            with self.assertRaisesRegex(ValueError, 'alpha/horizon'):
                v.load_refiner(args.checkpoint, .4, 2, 'cpu', 3)
            self.sample(root / 'val', 'train')
            with self.assertRaisesRegex(ValueError, 'overlap'):
                v.train_refiner(args)

    def test_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            obs = {'state.joint_position': np.zeros((1, 1, 7), np.float32),
                   'state.gripper_position': np.zeros((1, 1, 1), np.float32)}
            np.savez(source / 'sample.npz', **obs,
                     **{'target.joint_position': np.ones((1, 2, 7)),
                        'target.gripper_position': np.ones((1, 2, 1))},
                     mask=np.ones((1, 2)), episode_id=np.array(['unique']))
            class Policy:
                alpha, horizon = .5, 2
                config_gr00t = {'state': {'delta_indices': [0]}}
                def get_action(self, observation):
                    return {'action.joint_position': np.zeros((1, 2, 7)),
                            'action.gripper_position': np.zeros((1, 2, 1))}, {}
            v.cache_samples(Policy(), source, root / 'cache')
            values, episodes = v.read_cache(root / 'cache', .5, 2)
            self.assertEqual(episodes, {'unique'})
            self.assertEqual(tuple(values['target'].shape), (1, 2, 8))
            torch.testing.assert_close(values['target'], torch.ones(1, 2, 8))

    def test_invalid_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.sample(path, mask=np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, 'mask'):
                v.read_cache(path, .5, 2)

    def test_refiner_called_after_blending(self):
        spec11 = importlib.util.spec_from_file_location('tests11', Path(__file__).with_name('test_main_v1_1.py'))
        tests11 = importlib.util.module_from_spec(spec11)
        spec11.loader.exec_module(tests11)
        policy = v.BlendedPolicy(tests11.FakeN17(), tests11.FakePi(), horizon=2)
        class Capture:
            def refine(self, actions, state):
                np.testing.assert_allclose(actions['action.joint_position'][0], 6)
                np.testing.assert_allclose(state[:, :7], .1)
                return actions
        policy.refiner = Capture()
        actions, info = policy.get_action(tests11.observation())
        self.assertEqual(info['learnable_network'], 'residual_flow')
        self.assertEqual(actions['action.joint_position'].shape, (2, 2, 7))


if __name__ == '__main__':
    unittest.main()
