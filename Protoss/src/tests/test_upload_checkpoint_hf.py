import json
import argparse
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

from src.upload_checkpoint_hf import upload, inventory


class UploadTests(unittest.TestCase):
    def test_privacy_complete_checkpoint_and_readback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("config.json", "model.safetensors", "optimizer.pt", "trainer_state.json"):
                (root / name).write_bytes(b"fixture")
            args = argparse.Namespace(folder=root, repo_id="user/model-a", workers=2, dry_run=False)
            api = Mock()
            api.model_info.return_value = types.SimpleNamespace(private=False)
            with self.assertRaises(ValueError):
                upload(args, api)
            api.upload_large_folder.assert_not_called()
            api.model_info.return_value = types.SimpleNamespace(private=True)
            api.list_repo_files.return_value = ["config.json", "model.safetensors"]
            with self.assertRaises(RuntimeError):
                upload(args, api)
            api.list_repo_files.return_value += ["optimizer.pt", "trainer_state.json"]
            upload(args, api)
            self.assertTrue(api.create_repo.call_args.kwargs["private"])
            self.assertEqual(api.upload_large_folder.call_args.kwargs["folder_path"], str(root.resolve()))
            api.reset_mock()
            args.dry_run = True
            upload(args, api)
            api.create_repo.assert_not_called()

    def test_incomplete_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(folder=root, repo_id="user/model", workers=2, dry_run=False)
            api = Mock()
            with self.assertRaises(ValueError):
                upload(args, api)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").touch()
            with self.assertRaises(ValueError):
                upload(args, api)
            api.create_repo.assert_not_called()

    def test_adapter_run_upload_includes_processor_and_loss_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "v16_model_fm_step1000"
            files = {
                "adapter/adapter_config.json": '{}',
                "adapter/adapter_model.safetensors": 'weights',
                "processor/processor_config.json": '{}',
                "processor/statistics.json": '{}',
                "inference/adapter_inference.json": json.dumps({"format": "protoss-peft-v1",
                    "base_model": "/old/pretrained", "adapter": "../adapter", "processor": "../processor"}),
                "training_loss.csv": 'step,loss\n1000,0.1\n',
                "checkpoint-1000/trainer_state.json": '{"log_history":[{"step":1000,"loss":0.1}]}',
            }
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            args = argparse.Namespace(folder=root, repo_id="user/adapter", workers=2, dry_run=False)
            api = Mock()
            api.model_info.return_value = types.SimpleNamespace(private=True)
            api.list_repo_files.return_value = list(files)
            upload(args, api)
            self.assertEqual(api.upload_large_folder.call_args.kwargs["folder_path"], str(root.resolve()))
            with self.assertRaisesRegex(ValueError, "complete run folder"):
                inventory(root / "inference")
            (root / "processor/statistics.json").unlink()
            with self.assertRaisesRegex(ValueError, "normalization statistics missing"):
                inventory(root)

    def test_stale_large_upload_cache_is_repaired_at_exact_remote_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ("config.json", "model.safetensors", "checkpoint-1000/trainer_state.json")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            args = argparse.Namespace(folder=root, repo_id="user/new-destination", workers=2, dry_run=False)
            api = Mock()
            api.model_info.return_value = types.SimpleNamespace(private=True)
            # The large uploader returns successfully but files only exist under
            # an obsolete prefix, as with reused metadata or a previous layout.
            remote = {"inference/" + name for name in names}
            api.list_repo_files.side_effect = lambda **kwargs: sorted(remote)
            def direct_upload(**kwargs):
                self.assertEqual(kwargs["repo_id"], args.repo_id)
                self.assertEqual(Path(kwargs["path_or_fileobj"]).read_bytes(), b"fixture")
                remote.add(kwargs["path_in_repo"])
            api.upload_file.side_effect = direct_upload
            upload(args, api)
            self.assertEqual({call.kwargs["path_in_repo"] for call in api.upload_file.call_args_list}, set(names))
            self.assertTrue(set(names).issubset(remote))
            self.assertEqual(api.list_repo_files.call_count, 2)
            api.delete_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
