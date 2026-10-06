import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import train_groot_styles_v1_2 as styles


class StyleTests(unittest.TestCase):
    def records(self):
        return [{"source_episode": task * 10 + i, "task": str(task),
                 "speed_rms_rad_s": float(i + 1), "accel_rms_rad_s2": float(i + 1)}
                for task in range(2) for i in range(6)]

    def test_physical_time_units(self):
        time = np.arange(10) / 15
        q = np.repeat((2 * time)[:, None], 7, axis=1)
        result = styles.motion_metrics(q, 15)
        self.assertAlmostEqual(result["speed_rms_rad_s"], 2)
        self.assertAlmostEqual(result["accel_rms_rad_s2"], 0)
        self.assertAlmostEqual(result["jerk_rms_rad_s3"], 0)
        quadratic = np.repeat((time ** 2)[:, None], 7, axis=1)
        self.assertAlmostEqual(styles.motion_metrics(quadratic, 15)["accel_rms_rad_s2"], 2)

    def test_task_balance_and_test_isolation(self):
        splits, skipped = styles.select_styles(self.records(), 42)
        self.assertEqual(skipped, [])
        self.assertEqual(splits, styles.select_styles(self.records(), 42)[0])
        ids = [{r["source_episode"] for r in splits[k]} for k in ("a", "b", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        for task in ("0", "1"):
            a = [r for r in splits["a"] if r["task"] == task]
            b = [r for r in splits["b"] if r["task"] == task]
            self.assertEqual(len(a), len(b))
            self.assertLess(np.mean([r["speed_rms_rad_s"] for r in a]),
                            np.mean([r["speed_rms_rad_s"] for r in b]))

    def test_retiming_needs_no_repeated_instruction(self):
        records = [{"source_episode": i, "task": f"unique instruction {i}"} for i in range(300)]
        splits = styles.retimed_splits(records, 42)
        self.assertEqual(splits, styles.retimed_splits(records, 42))
        self.assertEqual([len(splits[k]) for k in ("a", "b", "test")], [240, 240, 60])
        train_ids = {r["source_episode"] for r in splits["a"]}
        test_ids = {r["source_episode"] for r in splits["test"]}
        self.assertEqual(train_ids, {r["source_episode"] for r in splits["b"]})
        self.assertFalse(train_ids & test_ids)
        # Editing A's transform metadata cannot silently modify B's record.
        splits["a"][0]["speed_factor"] = .75
        self.assertNotIn("speed_factor", splits["b"][0])

    def test_insufficient_or_indistinguishable_styles_fail(self):
        with self.assertRaises(ValueError):
            styles.select_styles(self.records()[:4], 42)
        records = self.records()
        for r in records:
            r["speed_rms_rad_s"] = r["accel_rms_rad_s2"] = 1
        with self.assertRaises(ValueError):
            styles.select_styles(records, 42)
        with self.assertRaises(ValueError):
            styles.motion_metrics(np.full((10, 7), np.nan), 15)

    def test_v3_to_v2_conversion_contract(self):
        # Exercise actual parquet/metadata conversion, mock only GR00T and video encoder.
        modules = {}
        for name in ("gr00t", "gr00t.data", "gr00t.data.state_action",
                     "gr00t.data.state_action.droid_frame", "gr00t.data.stats",
                     "gr00t.data.embodiment_tags"):
            modules[name] = types.ModuleType(name)
        modules["gr00t.data.state_action.droid_frame"].compute_eef_9d = lambda q: np.pad(q, ((0, 0), (0, 3)))
        def stats(root):
            (root / "meta/stats.json").write_text("{}")
        def rel_stats(root, tag):
            (root / "meta/relative_stats.json").write_text("{}")
        modules["gr00t.data.stats"].generate_stats = stats
        modules["gr00t.data.stats"].generate_rel_stats = rel_stats
        modules["gr00t.data.embodiment_tags"].EmbodimentTag = types.SimpleNamespace(resolve=lambda tag: tag)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            path = source / "data/chunk-000/file-000.parquet"
            path.parent.mkdir(parents=True)
            rows, frames = [], []
            for i in range(6):
                row = {"episode_index": i, "length": 8, "data/chunk_index": 0, "data/file_index": 0}
                for camera in styles.CAMERAS.values():
                    for key, value in {"chunk_index": 0, "file_index": 0,
                                       "from_timestamp": i, "to_timestamp": i + 8 / 15}.items():
                        row[f"videos/{camera}/{key}"] = value
                rows.append(row)
                for t in range(8):
                    frame = {"episode_index": i, "frame_index": t, "timestamp": t / 15,
                             "language_instruction": "pick block"}
                    for prefix in ("observation.state", "action"):
                        frame[prefix + ".joint_position"] = [((i + 1) * t ** 2) / 1000] * 7
                        frame[prefix + ".gripper_position"] = [0.5]
                        frame[prefix + ".cartesian_position"] = [0.] * 6
                    frames.append(frame)
            pd.DataFrame(frames).to_parquet(path, index=False)
            (source / "subset.json").write_text(json.dumps({"fps": 15, "revision": "fixture", "episodes": rows}))
            args = argparse.Namespace(dataset_dir=source, output_dir=root / "out", seed=42, style_method="select")
            def encode(command, **kwargs):
                Path(command[-1]).write_bytes(b"mock-video")
            with patch.dict(sys.modules, modules), patch.object(styles, "decode_episode", return_value=np.zeros((8, 180, 320, 3), np.uint8)), patch.object(styles.subprocess, "run", side_effect=encode):
                styles.prepare(args)
            report = json.loads((args.output_dir / "styles.json").read_text())
            for split in ("a", "b", "test"):
                folder = args.output_dir / split
                info = json.loads((folder / "meta/info.json").read_text())
                self.assertEqual(info["total_episodes"], len(report["splits"][split]))
                self.assertEqual(info["fps"], 15)
                global_indices = []
                for parquet in sorted(folder.glob("data/*/*.parquet")):
                    frame = pd.read_parquet(parquet)
                    self.assertEqual(np.stack(frame["action"]).shape, (8, 17))
                    np.testing.assert_array_equal(np.stack(frame["action"])[:, 9], .5)
                    global_indices.extend(frame["index"])
                self.assertEqual(global_indices, list(range(info["total_frames"])))
                self.assertTrue((folder / "meta/relative_stats.json").exists())
            with self.assertRaises(ValueError):
                styles.prepare(args)
            args.output_dir = root / "retimed"
            args.style_method = "retime"
            args.slow_factor, args.fast_factor, args.smooth_window = .75, 1.25, 5
            video_lengths = []
            def encode_retimed(command, **kwargs):
                video_lengths.append(len(kwargs["input"]) // (180 * 320 * 3))
                encode(command, **kwargs)
            with patch.dict(sys.modules, modules), patch.object(styles, "decode_episode", return_value=np.zeros((8, 180, 320, 3), np.uint8)), patch.object(styles.subprocess, "run", side_effect=encode_retimed):
                styles.prepare(args)
            report = json.loads((args.output_dir / "styles.json").read_text())
            a_ids = {r["source_episode"] for r in report["splits"]["a"]}
            b_ids = {r["source_episode"] for r in report["splits"]["b"]}
            test_ids = {r["source_episode"] for r in report["splits"]["test"]}
            self.assertEqual(a_ids, b_ids)
            self.assertFalse(a_ids & test_ids)
            self.assertTrue(report["synthetic"])
            expected_lengths = []
            for split, count in (("a", 10), ("b", 7), ("test", 8)):
                for parquet in sorted((args.output_dir / split).glob("data/*/*.parquet")):
                    frame = pd.read_parquet(parquet)
                    self.assertEqual(len(frame), count)
                    np.testing.assert_allclose(frame["timestamp"], np.arange(count) / 15)
                    np.testing.assert_array_equal(frame["frame_index"], np.arange(count))
                    expected_lengths.extend([count, count])
            self.assertEqual(video_lengths, expected_lengths)

    def test_retime_alignment_speed_and_rotation_wrap(self):
        n = 61
        rows = []
        for i in range(n):
            row = {"language_instruction": "task"}
            for prefix in ("observation.state", "action"):
                row[prefix + ".joint_position"] = [i / 15] * 7
                row[prefix + ".gripper_position"] = [float(i >= 30)]
                row[prefix + ".cartesian_position"] = [i / 15, 0, 0, 0, 0, np.deg2rad(179 if i < 30 else -179)]
            rows.append(row)
        source = pd.DataFrame(rows)
        for factor in (.75, 1.25):
            output, nearest = styles.retime_frame(source, 15, factor, 1)
            joints = np.stack(output["action.joint_position"])
            self.assertAlmostEqual(styles.motion_metrics(joints, 15)["speed_rms_rad_s"], factor, places=5)
            np.testing.assert_array_equal(joints[[0, -1]], np.stack(source["action.joint_position"])[[0, -1]])
            np.testing.assert_array_equal(np.stack(output["action.gripper_position"])[:, 0], nearest >= 30)
            # Rotations cross +/-pi using the short arc, rather than passing through zero.
            self.assertTrue((np.abs(np.stack(output["action.cartesian_position"])[:, 5]) > 3).all())
        noisy = np.sin(np.arange(101) * .1)[:, None] + .1 * (-1.) ** np.arange(101)[:, None]
        filtered = styles.smooth_positions(noisy, 5)
        self.assertLess(np.square(np.diff(filtered, n=2, axis=0)).mean(), np.square(np.diff(noisy, n=2, axis=0)).mean())
        np.testing.assert_allclose(filtered[[0, -1]], noisy[[0, -1]])

    def test_independent_training_uses_original_base_and_exact_output(self):
        captured = []
        class Config:
            def __init__(self):
                self.data = types.SimpleNamespace()
                self.model = types.SimpleNamespace()
                self.training = types.SimpleNamespace()
            def load_dict(self, value):
                self.data.datasets = value["data"]["datasets"]
                return self
        modules = {name: types.ModuleType(name) for name in
                   ("torch", "gr00t", "gr00t.configs", "gr00t.configs.base_config",
                    "gr00t.data", "gr00t.data.embodiment_tags", "gr00t.experiment",
                    "gr00t.experiment.experiment")}
        modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: True)
        modules["gr00t.configs.base_config"].get_default_config = Config
        modules["gr00t.data.embodiment_tags"].EmbodimentTag = types.SimpleNamespace(
            resolve=lambda tag: types.SimpleNamespace(value=tag.lower()))
        modules["gr00t.experiment.experiment"].run = captured.append
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules):
            root = Path(tmp).resolve()
            styles.write_json(root / "styles.json", {"splits": {"a": [1], "b": [2], "test": [3]}})
            for style in ("a", "b"):
                (root / style / "meta").mkdir(parents=True)
                (root / style / "meta/relative_stats.json").write_text("{}")
                args = argparse.Namespace(styles_dir=root, style=style, base_model=root / "original",
                         output_dir=root / ("model_" + style), seed=42, batch_size=2,
                         gradient_accumulation=16, lr=1e-5, max_steps=2000, save_steps=500,
                         workers=2, resume=False)
                styles.train(args)
                # Refuse accidental rerun/overwrite, before calling the trainer.
                with self.assertRaises(ValueError):
                    styles.train(args)
            self.assertEqual(len(captured), 2)
            for i, style in enumerate(("a", "b")):
                cfg = captured[i]
                self.assertEqual(cfg.training.start_from_checkpoint, str(root / "original"))
                self.assertEqual(cfg.training.output_dir, str(root / ("model_" + style)))
                self.assertIsNone(cfg.training.experiment_name)
                self.assertEqual(cfg.data.datasets[0]["dataset_paths"], [str(root / style)])
                self.assertFalse(cfg.model.tune_llm or cfg.model.tune_visual)
                self.assertTrue(cfg.model.tune_projector and cfg.model.tune_diffusion_model)


if __name__ == "__main__":
    unittest.main()
