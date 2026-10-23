"""Exercise the coordinator with real child processes and a ZeroMQ test server."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from src import training_evaluation as pipeline
from src.robolab_results import process_results
from src.robolab_testset import TEST_TASKS


class TrainingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tools = self.root / "tools"
        self.tools.mkdir()
        parser = argparse.ArgumentParser()
        pipeline.add_evaluation_args(parser)
        self.args = parser.parse_args([])
        self.args.output_dir = self.root / "model"
        self.args.base_model = self.root / "base"
        self.args.groot_root = self.root / "groot"
        self.args.eval_stack_root = self.root / "stack"
        self.args.eval_server_timeout = 5
        self.args.no_hf_upload = True
        self.args.hf_repo_id = None
        self.arguments = ["--output-dir", str(self.args.output_dir)]
        self.patch_tools = patch.object(pipeline, "TOOLS", self.tools)
        self.patch_tools.start()
        self.addCleanup(self.patch_tools.stop)
        self.environment = patch.dict(os.environ, {"PROTOSS_TEST_MODE": "ok"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        (self.tools / "train.py").write_text('''
import json, os, pathlib, sys
assert "--no-eval" in sys.argv
root = pathlib.Path(sys.argv[sys.argv.index("--output-dir") + 1])
if os.environ["PROTOSS_TEST_MODE"] == "training_failure": sys.exit(7)
root.mkdir()
(root / "training.pid").write_text(str(os.getpid()))
(root / "adapter").mkdir()
(root / "adapter/adapter_config.json").write_text("{}")
(root / "adapter/adapter_model.safetensors").write_text("test")
(root / "processor").mkdir()
(root / "processor/processor_config.json").write_text("{}")
(root / "lora_settings.json").write_text(json.dumps({"backend": "peft"}))
''')
        (self.tools / "serve.py").write_text('''
import json, os, pathlib, sys, time
import msgpack, zmq
root = pathlib.Path(sys.argv[sys.argv.index("--model-path") + 1])
mode = os.environ["PROTOSS_TEST_MODE"]
if mode == "remote_only":
    # Stand in for serve.py downloading an absent HF run before model loading.
    assert "--hf-repo-id" in sys.argv
    assert not (root / "adapter").exists()
    (root / "adapter").mkdir()
    (root / "adapter/adapter_config.json").write_text("{}")
    (root / "adapter/adapter_model.safetensors").write_text("downloaded")
    (root / "processor").mkdir()
    (root / "processor/processor_config.json").write_text("{}")
    (root / "lora_settings.json").write_text(json.dumps({"backend": "peft"}))
else:
    try:
        os.kill(int((root / "training.pid").read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError("Training process still owns resources")
(root / "server.pid").write_text(str(os.getpid()))
(root / "server_args.json").write_text(json.dumps(sys.argv))
if mode == "server_failure": sys.exit(8)
if mode == "server_timeout": time.sleep(60)
port = int(sys.argv[sys.argv.index("--port") + 1])
with zmq.Context() as context:
    with context.socket(zmq.REP) as server:
        server.bind(f"tcp://127.0.0.1:{port}")
        while True:
            request = msgpack.unpackb(server.recv(), raw=False)
            assert request["endpoint"] == "ping"
            server.send(msgpack.packb({"status": "ok"}))
''')
        (self.tools / "client.py").write_text('''
import json, os, pathlib, sys
mode = os.environ["PROTOSS_TEST_MODE"]
if mode == "preflight_failure" and sys.argv[1] == "preflight": sys.exit(9)
if sys.argv[1] == "preflight": sys.exit(0)
if mode == "client_failure": sys.exit(10)
name = sys.argv[sys.argv.index("--output-folder-name") + 1]
root = pathlib.Path(os.environ["ROBOLAB_595_ROOT"]) / "RoboLab/output" / name
root.mkdir(parents=True)
tasks = sys.argv[sys.argv.index("--task") + 1:sys.argv.index("--num-envs")]
num_envs = int(sys.argv[sys.argv.index("--num-envs") + 1])
num_runs = int(sys.argv[sys.argv.index("--num-runs") + 1])
rows = [{"task_name": task, "success": True, "score": 1,
         "run_idx": run_idx, "env_id": env_id, "metrics": {"speed": 2, "path_length": 3}}
        for task in tasks for run_idx in range(num_runs) for env_id in range(num_envs)]
(root / "episode_results.jsonl").write_text("\\n".join(json.dumps(row) for row in rows) + "\\n")
''')
        (self.tools / "setup_robolab_595.sh").write_text(
            f'#!/bin/bash\nexec "{sys.executable}" "{self.tools / "client.py"}" "$@"\n')

    def run_pipeline(self, mode="ok", repo=None, api=None):
        os.environ["PROTOSS_TEST_MODE"] = mode
        def calculate(*args, **kwargs):
            return process_results(*args, **kwargs, api=api)
        with patch("src.hf_model_sync.training_destination", return_value=repo), \
                patch("src.robolab_results.process_results", side_effect=calculate):
            pipeline.train_and_evaluate(self.args, self.arguments)

    def assert_server_stopped(self):
        pid = int((self.args.output_dir / "server.pid").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_full_local_pipeline_releases_training_and_saves_metrics(self):
        self.run_pipeline()
        status = json.loads((self.args.output_dir / "evaluation_status.json").read_text())
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["success_rate"], 1.)
        metrics = self.args.output_dir / "evaluations" / status["run_name"]
        summary = json.loads((metrics / "metrics_summary.json").read_text())
        self.assertEqual(summary["overall"]["metric_averages"]["path_length"], 3)
        self.assertTrue((metrics / "evaluation_config.json").is_file())
        self.assertEqual(status['tasks'], list(TEST_TASKS))
        self.assertEqual((status['num_envs'], status['num_runs']), (1, 10))
        self.assertEqual(len(summary['tasks']), 5)
        self.assertEqual([task['task_name'] for task in summary['tasks']], list(TEST_TASKS))
        self.assertTrue(all(task['records'] == 10 for task in summary['tasks']))
        self.assertEqual(summary['overall']['records'], 50)
        self.assertEqual(len((metrics / 'episode_results.jsonl').read_text().splitlines()), 50)
        self.assert_server_stopped()

    def test_hf_destination_download_and_metric_publication_are_same(self):
        api = Mock()
        self.run_pipeline(repo="account/experiment", api=api)
        arguments = json.loads((self.args.output_dir / "server_args.json").read_text())
        self.assertEqual(arguments[arguments.index("--hf-repo-id") + 1], "account/experiment")
        self.assertEqual(api.upload_folder.call_args.kwargs["repo_id"], "account/experiment")
        self.assertTrue(api.upload_folder.call_args.kwargs["path_in_repo"].startswith("evaluations/eval-model-"))
        self.assert_server_stopped()

    def test_standalone_hf_evaluation_can_download_an_absent_local_run(self):
        self.assertFalse(self.args.output_dir.exists())
        os.environ['PROTOSS_TEST_MODE'] = 'remote_only'
        api = Mock()
        with patch('src.robolab_results.process_results', side_effect=lambda *a, **kw: process_results(*a, **kw, api=api)):
            pipeline.evaluate_trained_run(self.args, 'account/remote-model')
        self.assertEqual((self.args.output_dir / 'adapter/adapter_model.safetensors').read_text(), 'downloaded')
        self.assertFalse((self.args.output_dir / 'training.pid').exists())
        status = json.loads((self.args.output_dir / 'evaluation_status.json').read_text())
        self.assertEqual(status['status'], 'complete')
        api.upload_folder.assert_called_once()
        self.assertEqual(api.upload_folder.call_args.kwargs['repo_id'], 'account/remote-model')
        self.assert_server_stopped()

    def test_local_only_evaluation_still_rejects_an_absent_run_before_starting_server(self):
        with patch.object(pipeline.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'requires an adapter run'):
                pipeline.evaluate_trained_run(self.args, None)
        start.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_preflight_failure_prevents_training(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline("preflight_failure")
        self.assertFalse(self.args.output_dir.exists())

    def test_training_failure_prevents_evaluation(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline("training_failure")
        self.assertFalse(self.args.output_dir.exists())

    def test_client_failure_prevents_metric_upload_and_stops_server(self):
        api = Mock()
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_pipeline("client_failure", "account/model", api)
        api.upload_folder.assert_not_called()
        self.assertEqual(json.loads((self.args.output_dir / "evaluation_status.json").read_text())["status"], "failed")
        self.assertTrue((self.args.output_dir / "adapter/adapter_model.safetensors").is_file())
        self.assert_server_stopped()

    def test_server_start_failure_prevents_client(self):
        with self.assertRaisesRegex(RuntimeError, "server exited"):
            self.run_pipeline("server_failure")
        self.assertFalse((self.args.eval_stack_root / "RoboLab/output").exists())
        self.assert_server_stopped()

    def test_server_timeout_stops_owned_process(self):
        self.args.eval_server_timeout = .8
        with self.assertRaises(TimeoutError):
            self.run_pipeline("server_timeout")
        self.assert_server_stopped()

    def test_metric_upload_failure_preserves_report_and_checkpoint(self):
        api = Mock()
        api.upload_folder.side_effect = RuntimeError("network error")
        with self.assertRaisesRegex(RuntimeError, "network error"):
            self.run_pipeline(repo="account/model", api=api)
        status = json.loads((self.args.output_dir / "evaluation_status.json").read_text())
        self.assertTrue((self.args.output_dir / "evaluations" / status["run_name"] / "metrics_summary.json").is_file())
        self.assert_server_stopped()


if __name__ == "__main__":
    unittest.main()
