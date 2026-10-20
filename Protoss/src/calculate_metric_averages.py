#!/usr/bin/env python3
"""Calculate average metrics from a RoboLab JSONL result file."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("RoboLab02_GR00T-N1.7-DROID.jsonl")

METRIC_ORDER = [
    "ee_sparc",
    "joint_sparc_mean",
    "ee_isj",
    "joint_isj",
    "ee_path_length",
    "joint_rmse_mean",
    "ee_speed_max",
    "ee_speed_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate numeric metric averages from a RoboLab JSONL file."
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"RoboLab JSONL result file (default: {DEFAULT_INPUT.name})",
    )
    return parser.parse_args()


def is_finite_number(value: Any) -> bool:
    """Return True for finite int/float values, excluding booleans."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def calculate_metric_averages(
    records: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    values: dict[str, list[float]] = defaultdict(list)

    for record in records:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for name, value in metrics.items():
            if is_finite_number(value):
                values[name].append(float(value))

    averages = {name: sum(items) / len(items) for name, items in values.items()}
    counts = {name: len(items) for name, items in values.items()}
    return averages, counts


def main() -> None:
    args = parse_args()
    path = args.jsonl_path.expanduser().resolve()
    records = load_records(path)
    averages, counts = calculate_metric_averages(records)

    successes = [record["success"] for record in records if isinstance(record.get("success"), bool)]
    scores = [float(record["score"]) for record in records if is_finite_number(record.get("score"))]

    print(f"File: {path}")
    print(f"Records: {len(records)}")
    if successes:
        success_count = sum(successes)
        print(f"Success rate: {success_count}/{len(successes)} ({success_count / len(successes):.2%})")
    if scores:
        print(f"Average score: {sum(scores) / len(scores):.6f} (n={len(scores)})")

    print("\nMetric averages:")
    if not averages:
        print("  No finite numeric metrics found.")
        return

    ordered_names = [name for name in METRIC_ORDER if name in averages]
    ordered_names.extend(sorted(name for name in averages if name not in METRIC_ORDER))

    width = max(len(name) for name in ordered_names)
    for name in ordered_names:
        print(f"  {name:<{width}} : {averages[name]:.6f} (n={counts[name]})")


if __name__ == "__main__":
    main()
