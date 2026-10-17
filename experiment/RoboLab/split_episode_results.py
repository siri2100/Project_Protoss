#!/usr/bin/env python3
"""Split a RoboLab episode_results JSONL file into one file per task."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path


DEFAULT_SUFFIX = "_GR00T-N1.7-DROID.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a RoboLab JSONL file by task_name (or env_name), preserving "
            "the order in which tasks first appear."
        )
    )
    parser.add_argument("input_file", type=Path, help="Combined episode_results JSONL file")
    parser.add_argument("start_number", type=int, help="First output number, e.g. 3")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: directory containing input_file)",
    )
    parser.add_argument("--prefix", default="RoboLab", help="Output filename prefix")
    parser.add_argument(
        "--suffix",
        default=DEFAULT_SUFFIX,
        help=f"Output filename suffix (default: {DEFAULT_SUFFIX})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist",
    )
    args = parser.parse_args()

    if args.start_number < 0:
        parser.error("start_number must be zero or greater")
    return args


def group_lines_by_task(input_file: Path) -> OrderedDict[str, list[str]]:
    groups: OrderedDict[str, list[str]] = OrderedDict()

    with input_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {input_file}:{line_number}: {exc}") from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {input_file}:{line_number}")

            task_name = record.get("task_name") or record.get("env_name")
            if not isinstance(task_name, str) or not task_name:
                raise ValueError(
                    f"Missing task_name and env_name at {input_file}:{line_number}"
                )

            # Preserve the original JSON line rather than serializing it again.
            groups.setdefault(task_name, []).append(line.rstrip("\r\n"))

    if not groups:
        raise ValueError(f"No records found in {input_file}")
    return groups


def main() -> None:
    args = parse_args()
    input_file = args.input_file.expanduser().resolve()
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_file.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_lines_by_task(input_file)
    outputs: list[tuple[Path, str, list[str]]] = []
    for number, (task_name, lines) in enumerate(groups.items(), start=args.start_number):
        destination = output_dir / f"{args.prefix}{number:02d}{args.suffix}"
        outputs.append((destination, task_name, lines))

    existing = [path for path, _, _ in outputs if path.exists()]
    if existing and not args.overwrite:
        paths = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            f"Output file(s) already exist. Use --overwrite to replace them:\n{paths}"
        )

    for destination, task_name, lines in outputs:
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{destination}: {task_name} ({len(lines)} records)")

    print(f"Created {len(outputs)} files from {sum(len(lines) for _, _, lines in outputs)} records.")


if __name__ == "__main__":
    main()
