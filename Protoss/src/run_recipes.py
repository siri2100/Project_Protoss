"""Dispatch the short commands documented in readme_train.md/readme_test.md."""

import argparse
import importlib.util
from pathlib import Path
import subprocess

from . import experiment_recipes as experiments
from . import setup_recipes as setup
from .recipe_runtime import Recipe

COMMANDS = {
    "setup-system": setup.setup_system,
    "update-project": setup.update_project,
    "folders": setup.folders,
    "setup-groot": setup.setup_groot,
    "login": setup.login,
    "setup-data": setup.setup_data,
    "download-droid": setup.download_droid,
    "prepare-data": setup.prepare_data,
    "setup-robolab": setup.setup_robolab,
    "smoke-robolab": setup.smoke_robolab,
    "setup-robolab51": setup.setup_robolab51,
    "patch-client51": setup.patch_client51,
    "server": experiments.server,
    "server-baseline": experiments.server_baseline,
    "client": experiments.client,
    "evaluate": experiments.evaluate,
    "metrics": experiments.metrics,
    "upload-model": experiments.upload_model,
    "upload-data": experiments.upload_data,
    "download-data": experiments.download_data,
    "check": experiments.check,
    "plot-distribution": experiments.plot_distribution,
}
for variant in ("FM", "S", "E"):
    COMMANDS["train-" + variant.lower()] = lambda ctx, extra, v=variant: experiments.train(ctx, v, extra)
    COMMANDS["resume-" + variant.lower()] = lambda ctx, extra, v=variant: experiments.resume(ctx, v, extra)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 Protoss/run.py train-s --name my_model_s --steps 2000 --lr 1e-6
  python3 Protoss/run.py train-e --steps 1000 --save-steps 250 --e-path-length-weight 0.02
  python3 Protoss/run.py server --model-name my_model_s --checkpoint-step 500
  python3 Protoss/run.py client --name my_model_s --num-runs 1

Original script options (loss weights, --no-eval, etc.) can be passed directly.
Explicit arguments override run_config.py defaults. The '--' separator is optional.""")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, help="Alternate Python config (default: Protoss/run_config.py)")
    parser.add_argument("--name", "--model-name", help="Training run/model folder name")
    parser.add_argument("--steps", "--step", "--max-steps", type=int,
                        help="Training/resume optimizer steps; overrides config and generated step suffix")
    parser.add_argument("--repo-id", help="HF destination/owner override: ACCOUNT/REPO")
    parser.add_argument("--port", type=int)
    parser.add_argument("--stack", choices=("595", "51"))
    parser.add_argument("--result-name", help="Manual evaluation result folder; default: timestamped")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without login, downloads or writes")
    args, extra = parser.parse_known_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if args.steps is not None:
        if not args.command.startswith(("train-", "resume-")):
            parser.error("--steps is for training/resume; use --checkpoint-step for server checkpoints")
        if args.steps < 1:
            parser.error("--steps must be positive")
        extra = ["--max-steps", str(args.steps), *extra]
    # Commands with no downstream argument parser must not silently ignore typos.
    no_extra = {"setup-system", "update-project", "folders", "setup-groot", "login", "setup-data",
                "setup-robolab", "setup-robolab51", "patch-client51"}
    if extra and args.command in no_extra:
        parser.error(f"Unsupported arguments for {args.command}: {' '.join(extra)}")
    config_path = (args.config or Path(__file__).resolve().parents[1] / "run_config.py").resolve()
    spec = importlib.util.spec_from_file_location("protoss_run_settings", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    ctx = Recipe(args, config)
    try:
        COMMANDS[args.command](ctx, extra)
    except subprocess.CalledProcessError as exc:
        print(f"Stopped: command failed (exit {exc.returncode}).")
        return exc.returncode if exc.returncode > 0 else 128 - exc.returncode
    except (OSError, ValueError) as exc:
        parser.exit(1, str(exc) + "\n")
    except KeyboardInterrupt:
        return 130
    return 0
