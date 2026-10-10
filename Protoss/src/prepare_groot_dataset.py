"""Prepare the shared original DROID train/test data for baseline / Model-FM/S/E."""
import argparse
from pathlib import Path
import sys

from .groot_common import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--style-method", choices=["original"], default="original")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.groot_root.resolve()))
    prepare(args)


if __name__ == "__main__":
    main()
