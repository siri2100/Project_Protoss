"""Train the controlled GR00T Model 1/S/E LoRA experiments."""
import argparse
import json
from pathlib import Path
import sys

from .groot_common import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="variant", choices=["1", "S", "E"], required=True)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--dataset-root", dest="styles_dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.lora_rank, args.max_steps, args.save_steps, args.batch_size, args.gradient_accumulation) < 1 or args.lr <= 0 or args.lora_alpha <= 0 or args.workers < 0:
        parser.error("Invalid positive training/LoRA parameters")
    report = json.loads((args.styles_dir / "styles.json").read_text())
    if report.get("synthetic") or report.get("style_method") != "original" or "train" not in report["splits"]:
        parser.error("Use the common original dataset prepared with --style-method original")
    args.style, args.lora = args.variant, True
    sys.path.insert(0, str(args.groot_root.resolve()))
    train(args)


if __name__ == "__main__":
    main()
