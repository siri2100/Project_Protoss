"""Rename RoboLab result files after an evaluation, preserving their contents."""

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--number", type=int, required=True, help="Experiment number (1 or greater)")
    parser.add_argument("--alpha", required=True, help="Same alpha as the Protoss server")
    args = parser.parse_args()
    try:
        alpha = Decimal(args.alpha)
    except InvalidOperation:
        parser.error("alpha must be a number between 0 and 1")
    if not alpha.is_finite() or not 0 <= alpha <= 1:
        parser.error("alpha must be a number between 0 and 1")
    if args.number < 1:
        parser.error("number must be 1 or greater")
    if not args.results_dir.is_dir():
        parser.error(f"results directory does not exist: {args.results_dir}")
    label = format(abs(alpha), ".2f")
    filename = f"RoboLab{args.number:02d}_alpha{label}.jsonl"
    sources = sorted(args.results_dir.rglob("episode_results.jsonl"))
    if not sources:
        parser.error("no episode_results.jsonl found; check the evaluation output directory")
    # Check every destination before changing any files.
    pairs = [(source, source.with_name(filename)) for source in sources]
    for source, target in pairs:
        if target.exists() or target.is_symlink():
            parser.error(f"refusing to overwrite existing result: {target}")
    for source, target in pairs:
        source.rename(target)
        print(target)


if __name__ == "__main__":
    main()
