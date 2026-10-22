"""Readable recipes around the existing training, serving and metric entrypoints."""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

from .recipe_runtime import options, save_launch


def train(ctx, variant, extra):
    _reject_managed(extra, ("--output-dir", "--model", "--groot-root", "--dataset-root", "--base-model"))
    settings = {**ctx.config.TRAIN_COMMON, **ctx.config.TRAIN[variant]}
    version = settings.pop("version")
    overrides = argparse.ArgumentParser(add_help=False)
    overrides.add_argument("--max-steps", type=int, default=settings["max_steps"])
    effective, _ = overrides.parse_known_args(extra)
    if effective.max_steps < 1:
        raise ValueError("max-steps must be positive")
    settings["max_steps"] = effective.max_steps
    name = ctx.args.name or f"{version}_model_{variant.lower()}_step{settings['max_steps']}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Training name must be a single folder name")
    output = ctx.checkpoints / name
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is nonempty: {output}. Use --name NEW_NAME or resume-{variant.lower()}.")
    arguments = ["--model", variant, "--groot-root", str(ctx.groot), "--dataset-root", str(ctx.data),
                 "--base-model", str(ctx.base), "--output-dir", str(output),
                 *options(settings), *ctx.evaluation_args()]
    if ctx.args.repo_id or ctx.config.HF_ACCOUNT:
        arguments += ["--hf-repo-id", ctx.repo(name)]
    arguments += extra
    # Keep launch records outside output: train.py requires an empty new output.
    save_launch(ctx, name, arguments)
    ctx.write(ctx.checkpoints / f"latest_{variant}_lora_v16.txt", str(output) + "\n")
    ctx.groot_run("train.py", arguments)


def resume(ctx, variant, extra):
    _reject_managed(extra, ("--output-dir", "--model", "--groot-root", "--dataset-root", "--base-model"))
    if ctx.args.name:
        output = ctx.model()
    else:
        pointer = ctx.checkpoints / f"latest_{variant}_lora_v16.txt"
        if ctx.args.dry_run:
            print(f"Resume uses saved command for {pointer}")
            return
        output = Path(pointer.read_text().strip())
    record = ctx.checkpoints / "launches" / f"{output.name}.json"
    if record.is_file():
        arguments = json.loads(record.read_text())["arguments"]
        if arguments[arguments.index("--model") + 1] != variant:
            raise ValueError("Saved launch belongs to a different model")
        # A checkout may have moved between training and resume (e.g. Project_Protoss2).
        arguments += ["--groot-root", str(ctx.groot), "--dataset-root", str(ctx.data),
                      "--base-model", str(ctx.base), "--output-dir", str(output)]
    elif not ctx.args.dry_run:
        arguments = _legacy_resume(ctx, output, variant)
    else:
        arguments = ["--model", variant, "--output-dir", str(output)]
    if ctx.args.repo_id:
        arguments += ["--hf-repo-id", ctx.args.repo_id]
    ctx.groot_run("train.py", [*arguments, "--resume", *extra])


def _legacy_resume(ctx, output, variant):
    """Recover existing README runs from their saved settings, not today's weights/LR."""
    lora = json.loads((output / "lora_settings.json").read_text())
    style = json.loads((output / "style_run.json").read_text())
    if lora.get("variant") not in (variant, "1" if variant == "FM" else variant):
        raise ValueError("Saved run belongs to a different model")
    settings = {**ctx.config.TRAIN_COMMON, **ctx.config.TRAIN[variant]}
    settings.pop("version")
    for key in ("batch_size", "lr", "max_steps", "gradient_accumulation", "seed"):
        settings[key] = style[key]
    settings.update(lora_rank=lora["rank"], lora_alpha=lora["alpha"])
    for key in list(settings):
        if key.startswith(("s_", "e_")):
            del settings[key]
    if variant == "S":
        if "sparc_weight" not in lora["model_s"]:
            raise ValueError("Old Model-S jerk runs cannot resume as SPARC; start a new run")
        for key, value in lora["model_s"].items():
            settings["s_" + key] = value
    if variant == "E":
        if lora.get("loss") != "pretrained_flow_preservation+fk_ee_path_length-v1":
            raise ValueError("Old Model-E joint-path runs cannot resume as FK EE paths; start a new run")
        settings.update(e_fmd_weight=lora["model_e"]["fm_weight"],
                        e_path_length_weight=lora["model_e"]["path_length_weight"])
    return ["--model", variant, "--groot-root", str(ctx.groot), "--dataset-root", str(ctx.data),
            "--base-model", str(ctx.base), "--output-dir", str(output),
            *options(settings), *ctx.evaluation_args()]


def server(ctx, extra):
    arguments = ["--groot-root", str(ctx.groot), "--hf-repo-id", ctx.repo(ctx.name()),
                 "--model-path", str(ctx.model()), "--base-model-path", str(ctx.base),
                 "--lora-mode", ctx.config.LORA_MODE,
                 "--embodiment-tag", "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                 "--device", "cuda:0", "--host", ctx.config.SERVER_HOST,
                 "--port", str(ctx.args.port or ctx.config.SERVER_PORT), "--use-sim-policy-wrapper"]
    if ctx.config.CHECKPOINT_STEP is not None:
        arguments += ["--checkpoint-step", str(ctx.config.CHECKPOINT_STEP)]
    ctx.groot_run("serve.py", [*arguments, *extra])


def server_baseline(ctx, extra):
    ctx.groot_run("serve.py", ["--groot-root", ctx.groot, "--model-path", ctx.base,
        "--embodiment-tag", "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT", "--device", "cuda:0",
        "--host", ctx.config.SERVER_HOST, "--port", str(ctx.args.port or ctx.config.SERVER_PORT),
        "--use-sim-policy-wrapper", *extra])


def evaluate(ctx, extra):
    arguments = ["--groot-root", ctx.groot, "--base-model", ctx.base,
                 "--output-dir", ctx.model(), *ctx.evaluation_args()]
    if "--no-hf-upload" not in extra:
        arguments += ["--hf-repo-id", ctx.repo(ctx.name())]
    ctx.groot_run("evaluate_robolab.py", [*arguments, *extra])


def client(ctx, extra):
    _reject_managed(extra, ("--output-folder-name", "--output_folder_name"))
    cfg = ctx.config.EVALUATION
    stack = ctx.args.stack or ctx.config.ROBOLAB_STACK
    if stack == "595":
        root = ctx.stack / "RoboLab"
        command = ["bash", ctx.tools / "setup_robolab_595.sh", "eval"]
    else:
        root = Path(ctx.config.ROBOLAB_51_ROOT)
        command = [root / ".venv-51/bin/python", "-u", root / "policies/gr00t/run.py"]
        ctx.run(["git", "apply", "--reverse", "--check", ctx.tools / "patches/robolab_fresh_results.patch"], cwd=root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    name = ctx.args.result_name or ctx.config.RESULT_NAME or f"{ctx.name()}-{stamp}"
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Result name must be a single folder name")
    results = root / "output" / name
    if results.exists():
        print(f"Previous results will be archived before a new evaluation: {results}", flush=True)
    repo = ctx.repo(ctx.name())
    arguments = ["--headless", "--device", cfg["device"], "--remote-host", ctx.config.SERVER_HOST,
                 "--remote-port", str(ctx.args.port or ctx.config.SERVER_PORT), "--task", *cfg["tasks"],
                 "--num-envs", str(cfg["num_envs"]), "--num-runs", str(cfg["num_runs"]),
                 "--open-loop-horizon", str(cfg["horizon"]), "--seed", str(cfg["seed"]),
                 "--instruction-type", "default", "--video-mode", "none", "--output-folder-name", name]
    ctx.run([*command, *arguments, *extra], cwd=root)
    # An exception above prevents both metric calculation and publication.
    ctx.groot_run("process_robolab_results.py", ["--results-dir", results,
        "--start-number", str(ctx.config.METRIC_START_NUMBER), "--hf-repo-id", repo])


def _reject_managed(extra, names):
    for token in extra:
        if token.split("=", 1)[0] in names:
            raise ValueError(f"{token}: use --name / --project-root / --result-name or run_config.py for managed paths")


def metrics(ctx, extra):
    status = ctx.model() / "evaluation_status.json"
    if ctx.args.result_name:
        root = ctx.stack / "RoboLab" if (ctx.args.stack or ctx.config.ROBOLAB_STACK) == "595" else Path(ctx.config.ROBOLAB_51_ROOT)
        results = root / "output" / ctx.args.result_name
    elif status.is_file():
        results = Path(json.loads(status.read_text())["results_dir"])
    elif ctx.args.dry_run:
        results = ctx.stack / "RoboLab/output/RESULT_NAME"
    else:
        raise FileNotFoundError("Pass --result-name or select a model with evaluation_status.json")
    ctx.groot_run("process_robolab_results.py", ["--results-dir", results,
        "--start-number", str(ctx.config.METRIC_START_NUMBER), "--hf-repo-id", ctx.repo(ctx.name()), *extra])


def upload_model(ctx, extra):
    ctx.groot_run("upload_checkpoint_hf.py", ["--folder", ctx.model(),
        "--repo-id", ctx.repo(ctx.name()), "--workers", "2", *extra])


def upload_data(ctx, extra):
    ctx.groot_run("upload_dataset_hf.py", ["--folder", ctx.data,
        "--repo-id", ctx.repo(ctx.config.DATASET_REPO), "--workers", "2", *extra])


def download_data(ctx, extra):
    from .setup_recipes import hf
    hf(ctx, ["download", ctx.repo(ctx.config.DATASET_REPO), "--repo-type", "dataset",
             "--local-dir", str(ctx.data), *extra])


def check(ctx, extra):
    ctx.run([ctx.python, "-m", "unittest", "discover", "-s", ctx.tools / "src/tests",
             "-t", ctx.tools, "-p", "test_*.py", "-v", *extra], cwd=ctx.groot)


def plot_distribution(ctx, extra):
    ctx.groot_run("plot_distribution.py", ["--groot-root", ctx.groot, "--base-model", ctx.base,
        "--model-path", ctx.model(), "--dataset-root", ctx.data, "--output-dir", ctx.model(), *extra])
