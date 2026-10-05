"""Synthetic LeRobot v3/FFmpeg integration: export -> cache -> train -> eval."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import prepare_droid_v1_2 as prepare

spec = importlib.util.spec_from_file_location('v12_pipeline', Path(__file__).with_name('main_v1.2.py'))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def config():
    return {'video': {'delta_indices': [-2, 0], 'modality_keys': list(prepare.CAMERAS)},
            'state': {'delta_indices': [0], 'modality_keys': ['joint_position', 'gripper_position', 'eef_9d']},
            'language': {'delta_indices': [0], 'modality_keys': ['annotation.language.language_instruction']},
            'action': {'delta_indices': list(range(40)), 'modality_keys': ['joint_position', 'gripper_position']}}


def eef(cart):
    correction = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    matrix = Rotation.from_euler('XYZ', cart[:, 3:]).as_matrix() @ correction
    return np.concatenate([cart[:, :3], matrix[:, :2].reshape(-1, 6)], axis=-1)


def make_fixture(root, count=10, length=12):
    root.mkdir(parents=True, exist_ok=True)
    rows, records = [], []
    for ep in range(count):
        row = {'episode_index': ep, 'length': length, 'tasks': ['pick block'],
               'data/chunk_index': 0, 'data/file_index': 0,
               'stats/is_episode_successful/min': [True]}
        for key in prepare.CAMERAS.values():
            row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                        f'videos/{key}/from_timestamp': ep * length / 15,
                        f'videos/{key}/to_timestamp': (ep + 1) * length / 15})
        rows.append(row)
        for t in range(length):
            records.append({'episode_index': ep, 'frame_index': t, 'timestamp': t / 15,
                            'language_instruction': 'pick block',
                            'observation.state.joint_position': [t / 100] * 7,
                            'observation.state.gripper_position': [0.],
                            'observation.state.cartesian_position': [0., 0., 0., .2, .3, .4],
                            'action.joint_position': [t / 100 + .1] * 7,
                            'action.gripper_position': [float(t % 2)]})
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(metadata)
    selected = prepare.select_episodes(metadata, count)
    manifest = {'repo_id': prepare.REPO, 'revision': 'fixture', 'fps': 15, 'episodes': selected}
    (root / 'subset.json').write_text(json.dumps(manifest))
    data = root / 'data/chunk-000/file-000.parquet'
    data.parent.mkdir(parents=True)
    pd.DataFrame(records).to_parquet(data)
    # Per-frame grayscale ramp, concatenated episodes with non-keyframe starts.
    video = np.zeros((count * length, 180, 320, 3), dtype=np.uint8)
    for i in range(len(video)):
        video[i] = i + 10
    for key in prepare.CAMERAS.values():
        path = root / prepare.video_filename(selected[0], key)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', '320x180', '-r', '15', '-i', 'pipe:0', '-an', '-threads', '1',
                        '-c:v', 'libx264', '-crf', '0', '-pix_fmt', 'yuv444p', str(path)],
                       input=video.tobytes(), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return rows


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg must be installed for video pipeline tests')
class DroidPipelineTests(unittest.TestCase):
    def test_export_cache_train_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_fixture(root / 'dataset')
            cfg = config()
            prepare.export_subset(root / 'dataset', root / 'export', cfg, eef, horizon=8, stride=4)
            manifest = json.loads((root / 'export/manifest.json').read_text())
            self.assertEqual([len(manifest['splits'][s]) for s in ('train', 'val', 'test')], [8, 1, 1])
            ids = [{r['episode'] for r in manifest['splits'][s]} for s in ('train', 'val', 'test')]
            self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
            ep = manifest['splits']['train'][0]['episode']
            sample = root / 'export/train' / f'episode_{ep:06d}/frame_000010.npz'
            with np.load(sample, allow_pickle=False) as values:
                self.assertEqual(values['mask'].sum(), 2)
                np.testing.assert_allclose(values['target.joint_position'][0, :2, 0], [.2, .21])
                video = values['video.exterior_image_1_left'][0]
                self.assertAlmostEqual(float(video[0].mean()), ep * 12 + 8 + 10, delta=2)
                self.assertAlmostEqual(float(video[1].mean()), ep * 12 + 10 + 10, delta=2)
                np.testing.assert_allclose(values['state.eef_9d'][0], eef(np.array([[0, 0, 0, .2, .3, .4]])), atol=1e-6)
            class Policy:
                alpha, horizon, config_gr00t = .5, 8, cfg
                calls = 0
                def get_action(self, obs):
                    self.calls += 1
                    state = v.current_state(obs, self.config_gr00t)
                    action = np.repeat(state[:, None], 8, axis=1)
                    return {'action.joint_position': action[..., :7], 'action.gripper_position': action[..., 7:]}, {}
            policy = Policy()
            for split in ('train', 'val', 'test'):
                v.cache_samples(policy, root/'export'/split, root/'cache'/split)
            calls = policy.calls
            v.cache_samples(policy, root/'export/train', root/'cache/train', resume=True)
            self.assertEqual(policy.calls, calls)
            args = argparse.Namespace(train_cache=root/'cache/train', val_cache=root/'cache/val',
                checkpoint=root/'model.pt', alpha=.5, horizon=8, hidden=16, batch_size=8,
                epochs=2, lr=.001, device='cpu', seed=42)
            v.train_refiner(args)
            args.test_cache, args.eval_seeds, args.flow_steps = root/'cache/test', [42, 43], 2
            args.metrics_output = root/'metrics.json'
            v.evaluate_refiner(args)
            result = json.loads(args.metrics_output.read_text())
            self.assertEqual(result['episodes'], 1)
            self.assertEqual(result['valid_timesteps'], 16)
            self.assertTrue(np.isfinite(result['v1.1_blend']['joint_mae_rad']))
            self.assertEqual(len(result['per_seed']), 2)
            args.test_cache = root/'cache/train'
            with self.assertRaisesRegex(ValueError, 'overlap'):
                v.evaluate_refiner(args)
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                prepare.export_subset(root/'dataset', root/'export', cfg, eef)

    def test_video_length_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = make_fixture(root, count=3)
            path = root / prepare.video_filename(rows[0], next(iter(prepare.CAMERAS.values())))
            with self.assertRaisesRegex(ValueError, 'decoded frames'):
                prepare.decode_episode(path, 0, 1000, 15)

    def test_fractional_timestamp_seek(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = make_fixture(root, count=3, length=11)
            key = next(iter(prepare.CAMERAS.values()))
            images = prepare.decode_episode(root / prepare.video_filename(rows[1], key), 11 / 15, 11, 15)
            self.assertAlmostEqual(float(images[0].mean()), 21, delta=2)
            self.assertAlmostEqual(float(images[-1].mean()), 31, delta=2)


class DataContractTests(unittest.TestCase):
    def test_train_eval_cli_without_model_servers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'test'):
                directory = root / split
                directory.mkdir()
                blend = np.zeros((2, 2, 8), dtype=np.float32)
                np.savez(directory / 'cache.npz', blend=blend, target=blend + .1,
                         state=np.zeros((2, 8), dtype=np.float32), mask=np.ones((2, 2)),
                         episode_id=np.array([split] * 2), alpha=.5, horizon=2)
            script = str(Path(__file__).with_name('main_v1.2.py'))
            shared = ['--checkpoint', str(root/'checkpoint.pt'), '--horizon', '2', '--device', 'cpu']
            result = subprocess.run([sys.executable, script, '--mode', 'train', *shared,
                '--train-cache', str(root/'train'), '--val-cache', str(root/'val'),
                '--epochs', '1', '--hidden', '16'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, script, '--mode', 'eval', *shared,
                '--test-cache', str(root/'test'), '--eval-seeds', '42', '43',
                '--metrics-output', str(root/'metrics.json')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads((root/'metrics.json').read_text())['per_seed']), 2)

    def test_success_language_filter_and_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for ep in range(5):
                row = {'episode_index': ep, 'tasks': ['' if ep == 0 else 'pick'], 'length': 10,
                       'data/chunk_index': 0, 'data/file_index': 0,
                       'stats/is_episode_successful/min': [ep != 1]}
                for key in prepare.CAMERAS.values():
                    for field in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp'):
                        row[f'videos/{key}/{field}'] = 0
                rows.append(row)
            path = root/'metadata.parquet'
            pd.DataFrame(rows).to_parquet(path)
            chosen = prepare.select_episodes(path, 3)
            self.assertEqual([r['episode_index'] for r in chosen], [2, 3, 4])
            splits = prepare.split_episodes(chosen, 42)
            self.assertEqual([len(splits[s]) for s in ('train', 'val', 'test')], [1, 1, 1])
            with self.assertRaisesRegex(ValueError, 'only 3'):
                prepare.select_episodes(path, 4)

    def test_masked_metrics(self):
        import torch
        prediction = torch.zeros(1, 2, 8)
        target = torch.ones(1, 2, 8)
        target[:, 1] = 10000
        result = v.action_metrics(prediction, target, torch.tensor([[1., 0.]]), torch.ones(8))
        self.assertEqual(result['joint_mae_rad'], 1.)
        self.assertEqual(result['normalized_action_mse'], 1.)


if __name__ == '__main__':
    unittest.main()
