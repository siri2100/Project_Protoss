import argparse
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

from src.upload_checkpoint_hf import upload
from src.upload_dataset_hf import inventory


class DatasetUploadTests(unittest.TestCase):
    def fixture(self, root):
        (root / "styles.json").write_text(json.dumps({"splits": {s: [{}] for s in ("train", "test")}}))
        for split in ("train", "test"):
            folder = root / {"train": "trainset", "test": "testset"}[split]
            (folder / "meta").mkdir(parents=True)
            info = {"total_episodes": 1, "total_frames": 4, "chunks_size": 1000,
                    "features": {"observation.images.wrist": {"dtype": "video"}},
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
            (folder / "meta/info.json").write_text(json.dumps(info))
            (folder / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": 4}) + "\n")
            for name in ("modality.json", "tasks.jsonl", "stats.json", "relative_stats.json"):
                (folder / "meta" / name).write_text("{}")
            for path in ("data/chunk-000/episode_000000.parquet",
                         "videos/chunk-000/observation.images.wrist/episode_000000.mp4"):
                target = folder / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"fixture")

    def test_dataset_type_privacy_readback_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / "source").mkdir()
            (root / "source/raw.parquet").write_bytes(b"raw")
            self.assertFalse(any(p.relative_to(root).parts[0] == "source" for p in inventory(root)))
            args = argparse.Namespace(folder=root, repo_id="user/styles", workers=2, dry_run=False)
            api = Mock()
            api.dataset_info.return_value = types.SimpleNamespace(private=True)
            api.list_repo_files.return_value = [p.relative_to(root).as_posix() for p in inventory(root)]
            upload(args, api, validator=inventory, repo_type="dataset")
            self.assertEqual(api.upload_large_folder.call_args.kwargs["repo_type"], "dataset")
            self.assertIn("source/**", api.upload_large_folder.call_args.kwargs["ignore_patterns"])
            self.assertTrue(api.create_repo.call_args.kwargs["private"])
            api.model_info.assert_not_called()
            api.reset_mock()
            args.dry_run = True
            upload(args, api, validator=inventory, repo_type="dataset")
            api.create_repo.assert_not_called()

    def test_incomplete_video_and_mismatched_episode_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            video = root / "testset/videos/chunk-000/observation.images.wrist/episode_000000.mp4"
            video.unlink()
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                inventory(root)
            video.write_bytes(b"fixture")
            report = root / "styles.json"
            data = json.loads(report.read_text())
            data["splits"]["train"] = []
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                inventory(root)


if __name__ == "__main__":
    unittest.main()
