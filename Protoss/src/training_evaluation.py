"""Sequence training, inference, RoboLab evaluation and metric publication.

The coordinator never loads a CUDA model. Training exits before inference starts.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time

from .robolab_testset import TEST_TASKS, TEST_NUM_ENVS, TEST_NUM_RUNS

TOOLS = Path(__file__).resolve().parents[1]


def _positive(value):
    import argparse
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _timeout(value):
    import argparse
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _port(value):
    import argparse
    number = int(value)
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError("must be between 0 (automatic) and 65535")
    return number


def _seed(value):
    import argparse
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32-1")
    return number


def add_evaluation_args(parser):
    group = parser.add_argument_group("Automatic RoboLab evaluation after training")
    group.add_argument("--no-eval", action="store_true", help="Train/upload only; skip RoboLab")
    group.add_argument("--eval-stack-root", type=Path,
                       default=Path(os.environ.get("ROBOLAB_595_ROOT", "/workspace/robolab-595")))
    group.add_argument("--eval-tasks", nargs="+", default=list(TEST_TASKS))
    group.add_argument("--eval-num-envs", type=_positive, default=TEST_NUM_ENVS)
    group.add_argument("--eval-num-runs", type=_positive, default=TEST_NUM_RUNS,
                       help=f"Sequential runs per task; total episodes per task = runs * envs (default: {TEST_NUM_RUNS} * {TEST_NUM_ENVS})")
    group.add_argument("--eval-horizon", type=_positive, default=8)
    group.add_argument("--eval-seed", type=_seed, default=0)
    group.add_argument("--eval-port", type=_port, default=0, help="Server port; 0 selects a free port")
    group.add_argument("--eval-device", default="cuda:0", help="Simulator device in CUDA_VISIBLE_DEVICES")
    group.add_argument("--eval-server-timeout", type=_timeout, default=600)


def _environment(args):
    env = dict(os.environ)
    for key in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(key, None)
    env["ROBOLAB_595_ROOT"] = str(args.eval_stack_root.expanduser().resolve())
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _free_port(port):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", port))
        return listener.getsockname()[1]


def preflight(args):
    """Check installed simulator patches and port before spending time training."""
    # Ping uses only msgpack/ZeroMQ; importing GR00T here would load torch/CUDA.
    import msgpack  # noqa: F401
    import zmq  # noqa: F401

    _free_port(args.eval_port)
    subprocess.run(["bash", str(TOOLS / "setup_robolab_595.sh"), "preflight"],
                   env=_environment(args), check=True)


def _ping(port):
    import msgpack
    import zmq

    with zmq.Context() as context:
        with context.socket(zmq.REQ) as request:
            request.setsockopt(zmq.LINGER, 0)
            request.setsockopt(zmq.SNDTIMEO, 500)
            request.setsockopt(zmq.RCVTIMEO, 500)
            request.connect(f"tcp://127.0.0.1:{port}")
            try:
                request.send(msgpack.packb({"endpoint": "ping"}))
                reply = msgpack.unpackb(request.recv(), raw=False)
            except zmq.Again:
                return False
            return isinstance(reply, dict) and reply.get("status") == "ok"


def _wait_for_server(server, port, timeout, log):
    deadline = time.monotonic() + timeout
    next_update = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Inference server exited ({server.returncode}). Log: {log}")
        if _ping(port):
            if server.poll() is not None:
                raise RuntimeError(f"Inference server exited during startup. Log: {log}")
            return
        if time.monotonic() >= next_update:
            print(f"Waiting for inference server; log: {log}", flush=True)
            next_update = time.monotonic() + 30
        time.sleep(1)
    raise TimeoutError(f"Inference server not ready after {timeout:g}s. Log: {log}")


def _stop_process(process):
    """Only stop a process group created by this coordinator."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_checked(command, env, server=None):
    """Propagate failure/interruption and clean up owned subprocesses."""
    child = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        while child.poll() is None:
            if server is not None and server.poll() is not None:
                raise RuntimeError("Inference server exited during RoboLab evaluation")
            time.sleep(.25)
        if child.returncode:
            raise subprocess.CalledProcessError(child.returncode, command)
    finally:
        _stop_process(child)


def evaluate_trained_run(args, destination):
    """Evaluate the final adapter; existing results are never resumed implicitly."""
    from .inference_loading import resolve_inference_paths
    from .robolab_results import process_results

    output = args.output_dir.expanduser().resolve()
    if destination:
        # serve.py downloads and validates the HF run/base before loading them.
        # A standalone evaluation may have no local model yet; create only the
        # directory needed for its server log instead of rejecting that run.
        output.mkdir(parents=True, exist_ok=True)
    else:
        resolve_inference_paths(output, args.base_model.resolve())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = re.sub(r"[^A-Za-z0-9._-]", "_", output.name)
    run_name = f"eval-{label}-{stamp}"
    results = args.eval_stack_root.expanduser().resolve() / "RoboLab/output" / run_name
    if results.exists():
        raise FileExistsError(f"Use a fresh evaluation directory: {results}")
    log_dir = output / "evaluation_logs"
    log_dir.mkdir(exist_ok=True)
    log = log_dir / f"{run_name}-server.log"
    env = _environment(args)
    port = _free_port(args.eval_port)
    command = [sys.executable, "-u", str(TOOLS / "serve.py"),
               "--groot-root", str(args.groot_root.resolve()),
               "--model-path", str(output), "--base-model-path", str(args.base_model.resolve()),
               "--lora-mode", "unmerged", "--device", "cuda:0", "--host", "127.0.0.1",
               "--port", str(port), "--use-sim-policy-wrapper",
               "--embodiment-tag", "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"]
    if destination:
        command += ["--hf-repo-id", destination]
    status = {"run_name": run_name, "repo_id": destination, "model_path": str(output),
              "results_dir": str(results), "tasks": args.eval_tasks,
              "num_envs": args.eval_num_envs, "num_runs": args.eval_num_runs,
              "horizon": args.eval_horizon, "seed": args.eval_seed, "port": port,
              "device": args.eval_device, "server_log": str(log), "status": "starting_server"}
    status_path = output / "evaluation_status.json"

    def save_status():
        status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    save_status()
    print(f"Starting evaluation server on port {port}; log: {log}", flush=True)
    try:
        with log.open("w", encoding="utf-8") as stream:
            server = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            try:
                _wait_for_server(server, port, args.eval_server_timeout, log)
                status["status"] = "evaluating"
                save_status()
                client = ["bash", str(TOOLS / "setup_robolab_595.sh"), "eval",
                          "--headless", "--device", args.eval_device, "--remote-host", "127.0.0.1",
                          "--remote-port", str(port), "--task", *args.eval_tasks,
                          "--num-envs", str(args.eval_num_envs), "--num-runs", str(args.eval_num_runs),
                          "--open-loop-horizon", str(args.eval_horizon), "--seed", str(args.eval_seed),
                          "--instruction-type", "default", "--video-mode", "none",
                          "--output-folder-name", run_name]
                _run_checked(client, env, server)
            finally:
                _stop_process(server)
        status["status"] = "processing_metrics"
        save_status()
        try:
            metadata = {key: value for key, value in status.items() if key != "status"}
            summary, _ = process_results(results, destination, run_name=run_name, metadata=metadata)
        finally:
            # Keep the reports beside checkpoints even if network publication fails.
            if (results / "metrics").is_dir():
                shutil.copytree(results / "metrics", output / "evaluations" / run_name)
        status["status"] = "complete"
        status["success_rate"] = summary["overall"]["success_rate"]
        status["average_score"] = summary["overall"]["average_score"]
        save_status()
        print(f"Training/evaluation complete. Metrics: {output / 'evaluations' / run_name}", flush=True)
    except BaseException as exc:
        status.update(status="failed", error=str(exc))
        save_status()
        print(f"Evaluation stopped; trained model retained. Status: {status_path}", flush=True)
        raise


def train_and_evaluate(args, training_arguments):
    from .hf_model_sync import training_destination

    preflight(args)
    destination = training_destination(args)
    worker = [sys.executable, "-u", str(TOOLS / "train.py"), *training_arguments, "--no-eval"]
    if destination:
        worker += ["--hf-repo-id", destination]
    print("Training worker starts; RoboLab runs after this process releases its GPU memory.", flush=True)
    _run_checked(worker, _environment(args))
    evaluate_trained_run(args, destination)
