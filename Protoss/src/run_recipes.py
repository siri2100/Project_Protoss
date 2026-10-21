"""Dispatch the short commands documented in readme_train.md/readme_test.md."""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, help="Alternate Python config (default: Protoss/run_config.py)")
    parser.add_argument("--name", help="Training run/model folder name")
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
    # Avoid accepting typos in recipe options; downstream flags go after '--'.
    actual = list(argv) if argv is not None else sys.argv[1:]
    if extra and "--" not in actual:
        parser.error("Pass underlying script options after '--', e.g. train-s -- --no-eval")
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
