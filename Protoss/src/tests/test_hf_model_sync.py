import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

from src.hf_model_sync import download_for_server, training_destination, upload_training_run


class ModelSyncTests(unittest.TestCase):
    def test_offline_training_skips_authentication(self):
        api = Mock()
        self.assertIsNone(training_destination(Namespace(no_hf_upload=True), api))
        api.whoami.assert_not_called()

    def test_download_failure_prevents_server_launch(self):
        args = Namespace(hf_repo_id="user/run", hf_revision="abc",
                         hf_base_repo_id="nvidia/base", hf_base_revision="def")
        downloader = Mock(side_effect=RuntimeError("network unavailable"))
        with tempfile.TemporaryDirectory() as tmp:
            remaining = ["--model-path", f"{tmp}/run", "--base-model-path", f"{tmp}/base"]
            with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                download_for_server(args, remaining, downloader)
        self.assertEqual(downloader.call_count, 1)

    def test_downloads_both_revisions_before_validation(self):
        args = Namespace(hf_repo_id="user/run", hf_revision="abc",
                         hf_base_repo_id="nvidia/base", hf_base_revision="def")
        downloader = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            remaining = ["--model-path", f"{tmp}/run", "--base-model-path", f"{tmp}/base",
                         "--port", "5557"]
            with patch("src.inference_loading.resolve_inference_paths", return_value=("base", "adapter", "processor")):
                self.assertEqual(download_for_server(args, remaining, downloader), remaining)
            self.assertEqual([c.kwargs["revision"] for c in downloader.call_args_list], ["abc", "def"])

    def test_upload_reuses_complete_run_validator(self):
        with patch("src.upload_checkpoint_hf.upload") as upload:
            upload_training_run(Path("run"), "user/run")
        args = upload.call_args.args[0]
        self.assertEqual(args.repo_id, "user/run")
        self.assertFalse(args.dry_run)
