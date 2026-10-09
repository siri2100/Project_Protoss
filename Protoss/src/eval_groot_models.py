"""Evaluate Model 0/1/S/E on the same original held-out data and seeds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from .dataset_paths import split_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["0", "1", "S", "E"], required=True)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Original pretrained model for 0; merged inference/ export for 1/S/E")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--traj-ids", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()
    report_path = args.dataset_root / "styles.json"
    report = json.loads(report_path.read_text())
    if report.get("style_method") != "original" or report.get("synthetic"):
        parser.error("Evaluation requires the common original dataset")
    if min(args.steps, args.execution_horizon, args.denoising_steps) < 1:
        parser.error("Steps and horizons must be positive")
    train_ids = {r["source_episode"] for r in report["splits"]["train"]}
    if train_ids & {r["source_episode"] for r in report["splits"]["test"]}:
        parser.error("Train/test source episode overlap")
    ids = args.traj_ids if args.traj_ids is not None else list(range(len(report["splits"]["test"])))
    if not ids or any(i < 0 or i >= len(report["splits"]["test"]) for i in ids):
        parser.error("Invalid test trajectory IDs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output is nonempty; use a new evaluation directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"model": args.model, "model_path": str(args.model_path.resolve()),
                "test_path": str(split_path(args.dataset_root, "test").resolve()), "traj_ids": ids,
                "seeds": args.seeds, "execution_horizon": args.execution_horizon,
                "steps": args.steps, "denoising_steps": args.denoising_steps,
                "dataset_manifest_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "metrics": "Native GR00T open-loop MSE/MAE; does not measure task duration/success"}
    (args.output_dir / "evaluation.json").write_text(json.dumps(settings, indent=2) + "\n")
    script = args.groot_root.resolve() / "gr00t/eval/open_loop_eval.py"
    # Set argv inside the child before invoking the upstream entry point.
    runner = "import runpy,sys; from transformers import set_seed; seed=int(sys.argv[1]); script=sys.argv[2]; sys.argv=sys.argv[2:]; set_seed(seed); runpy.run_path(script,run_name='__main__')"
    for seed in args.seeds:
        print(f"Evaluating Model {args.model}, seed {seed}; "
              f"progress log: {args.output_dir / f'seed_{seed}.log'}", flush=True)
        command = [sys.executable, "-c", runner, str(seed), str(script),
            "--dataset-path", settings["test_path"], "--model-path", settings["model_path"],
            "--embodiment-tag", "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            "--traj-ids", *map(str, ids), "--execution-horizon", str(args.execution_horizon),
            "--steps", str(args.steps), "--denoising-steps", str(args.denoising_steps),
            "--modality-keys", "joint_position", "gripper_position", "--save-plot-path",
            str((args.output_dir / f"seed_{seed}").resolve())]
        with (args.output_dir / f"seed_{seed}.log").open("w") as log:
            subprocess.run(command, cwd=args.groot_root.resolve(), stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Model {args.model} seed {seed} complete: {args.output_dir / f'seed_{seed}.log'}", flush=True)


if __name__ == "__main__":
    main()
