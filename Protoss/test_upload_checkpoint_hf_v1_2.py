import argparse
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

from upload_checkpoint_hf_v1_2 import upload


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


if __name__ == "__main__":
    unittest.main()
