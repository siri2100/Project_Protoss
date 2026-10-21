"""Run automatic server/client/metrics stages for a saved run, without training."""

import argparse
from pathlib import Path

from src.hf_model_sync import training_destination
from src.training_evaluation import add_evaluation_args, evaluate_trained_run, preflight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Saved training run")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--no-hf-upload", action="store_true")
    add_evaluation_args(parser)
    args = parser.parse_args()
    if args.no_eval:
        parser.error("--no-eval is only meaningful for train.py")
    preflight(args)
    evaluate_trained_run(args, training_destination(args))
