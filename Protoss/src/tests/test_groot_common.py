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
from src import groot_common as styles

class CommonTests(unittest.TestCase):
    def test_physical_time_units(self):
        time = np.arange(10) / 15
        q = np.repeat((2 * time)[:, None], 7, axis=1)
        result = styles.motion_metrics(q, 15)
        self.assertAlmostEqual(result["speed_rms_rad_s"], 2)
        self.assertAlmostEqual(result["accel_rms_rad_s2"], 0)
        self.assertAlmostEqual(result["jerk_rms_rad_s3"], 0)
        quadratic = np.repeat((time ** 2)[:, None], 7, axis=1)
        self.assertAlmostEqual(styles.motion_metrics(quadratic, 15)["accel_rms_rad_s2"], 2)

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
            args = argparse.Namespace(dataset_dir=source, output_dir=root / "out", seed=42, style_method="original")
            for name in ("trainset", "testset"):
                (args.output_dir / name).mkdir(parents=True)
                (args.output_dir / name / ".gitkeep").touch()
            def encode(command, **kwargs):
                Path(command[-1]).write_bytes(b"mock-video")
            with patch.dict(sys.modules, modules), patch.object(styles, "decode_episode", return_value=np.zeros((8, 180, 320, 3), np.uint8)), patch.object(styles.subprocess, "run", side_effect=encode):
                styles.prepare(args)
            report = json.loads((args.output_dir / "styles.json").read_text())
            for split in ("train", "test"):
                folder = styles.split_path(args.output_dir, split)
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
            args.output_dir = root / "shared"
            args.style_method = "original"
            with patch.dict(sys.modules, modules), patch.object(styles, "decode_episode", return_value=np.zeros((8, 180, 320, 3), np.uint8)), patch.object(styles.subprocess, "run", side_effect=encode):
                styles.prepare(args)
            report = json.loads((args.output_dir / "styles.json").read_text())
            self.assertEqual(set(report["splits"]), {"train", "test"})
            self.assertFalse(report["synthetic"])
            for split in ("train", "test"):
                for i, record in enumerate(report["splits"][split]):
                    frame = pd.read_parquet(styles.split_path(args.output_dir, split) / f"data/chunk-000/episode_{i:06d}.parquet")
                    expected = ((record["source_episode"] + 1) * np.arange(8) ** 2) / 1000
                    np.testing.assert_allclose(np.stack(frame["action"])[:, 10], expected)
                    self.assertEqual(len(frame), 8)

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
            styles.write_json(root / "styles.json", {"splits": {"train": [1, 2], "test": [3]}})
            (root / "trainset/meta").mkdir(parents=True)
            (root / "trainset/meta/relative_stats.json").write_text("{}")
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
                self.assertEqual(cfg.data.datasets[0]["dataset_paths"], [str(root / "trainset")])
                self.assertFalse(cfg.model.tune_llm or cfg.model.tune_visual)
                self.assertTrue(cfg.model.tune_projector and cfg.model.tune_diffusion_model)

    def test_shared_split_without_repeated_tasks(self):
        records = [{"source_episode": i, "task": str(i)} for i in range(300)]
        splits = styles.split_sources(records, 42)
        self.assertEqual(splits, styles.split_sources(records, 42))
        self.assertEqual([len(splits[k]) for k in ("train", "test")], [240, 60])
        self.assertFalse({r["source_episode"] for r in splits["train"]} & {r["source_episode"] for r in splits["test"]})
        with self.assertRaises(ValueError):
            styles.split_sources(records[:2], 42)

if __name__ == "__main__":
    unittest.main()
