"""Standard-library-only execution support for the README recipes."""

import json
import os
from pathlib import Path
import shlex
import subprocess


class Recipe:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.root = args.project_root.expanduser().resolve()
        self.tools = self.root / "Protoss"
        self.groot = self.root / config.GROOT_DIRECTORY
        self.python = self.groot / ".venv/bin/python"
        self.checkpoints = self.root.parent / "checkpoint"
        self.base = self.checkpoints / "GR00T-N1.7-DROID"
        self.data = self.tools / "data"
        self.stack = Path(config.ROBOLAB_STACK_ROOT).expanduser().resolve()
        self.env = dict(os.environ)
        for key in ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH"):
            self.env.pop(key, None)
        self.env.update(
            PATH=str(Path.home() / ".local/bin") + os.pathsep + self.env.get("PATH", ""),
            PROTOSS_ROOT=str(self.root), CHECKPOINT_ROOT=str(self.checkpoints),
            HF_HOME=config.HF_HOME, HF_HUB_ENABLE_HF_TRANSFER="0",
            CUDA_VISIBLE_DEVICES=config.CUDA_VISIBLE_DEVICES,
            ROBOLAB_595_ROOT=str(self.stack), PYTHONUNBUFFERED="1",
            UV_CACHE_DIR="/workspace/.cache/uv", UV_LINK_MODE="copy",
        )

    def run(self, command, cwd=None, env=None, capture=False):
        command = [str(item) for item in command]
        location = Path(cwd or self.root)
        print(f"[{location}] {shlex.join(command)}", flush=True)
        if self.args.dry_run:
            return ""
        result = subprocess.run(command, cwd=location, env=env or self.env,
                                check=True, text=True, stdout=subprocess.PIPE if capture else None)
        return result.stdout.strip() if capture else ""

    def groot_run(self, script, arguments=(), capture=False):
        if not self.args.dry_run and not self.python.is_file():
            raise FileNotFoundError(f"GR00T Python not found: {self.python}. Run setup-groot first.")
        return self.run([self.python, "-u", self.tools / script, *arguments],
                        cwd=self.groot, capture=capture)

    def python_code(self, code, capture=False):
        return self.run([self.python, "-c", code], cwd=self.groot, capture=capture)

    def mkdir(self, path):
        if not self.args.dry_run:
            Path(path).mkdir(parents=True, exist_ok=True)

    def write(self, path, text):
        if not self.args.dry_run:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    def repo(self, name):
        if self.args.repo_id:
            return self.args.repo_id
        account = self.config.HF_ACCOUNT
        if not account:
            if self.args.dry_run:
                account = "ACCOUNT"
            else:
                account = self.python_code(
                    'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])', capture=True)
        return f"{account}/{name}"

    def name(self):
        name = self.args.name or self.config.MODEL_NAME
        if not name or name in (".", "..") or Path(name).name != name:
            raise ValueError("--name must be a single folder name")
        return name

    def model(self):
        return self.checkpoints / self.name()

    def evaluation_args(self):
        cfg = self.config.EVALUATION
        return ["--eval-stack-root", str(self.stack), "--eval-tasks", *cfg["tasks"],
                "--eval-num-envs", str(cfg["num_envs"]), "--eval-num-runs", str(cfg["num_runs"]),
                "--eval-horizon", str(cfg["horizon"]), "--eval-seed", str(cfg["seed"]),
                "--eval-device", cfg["device"], "--eval-server-timeout", str(cfg["server_timeout"])]


def options(settings):
    """Turn readable Python settings into the existing CLI flags."""
    arguments = []
    for name, value in settings.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        elif value is not None:
            arguments.extend([flag, str(value)])
    return arguments


def save_launch(recipe, name, arguments):
    recipe.write(recipe.checkpoints / "launches" / f"{name}.json",
                 json.dumps({"arguments": list(map(str, arguments))}, indent=2) + "\n")
