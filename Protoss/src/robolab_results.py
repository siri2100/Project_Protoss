"""Server-side equivalent of experiment/README_exp.md, for every evaluated task."""

import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from .calculate_metric_averages import METRIC_ORDER, calculate_metric_averages, is_finite_number
from .split_episode_results import group_lines_by_task


def summarize(records):
    averages, counts = calculate_metric_averages(records)
    if not averages:
        raise ValueError("No finite numeric metrics found; check RoboLab's episode_results.jsonl")
    successes = [record["success"] for record in records if isinstance(record.get("success"), bool)]
    scores = [float(record["score"]) for record in records if is_finite_number(record.get("score"))]
    return {
        "records": len(records),
        "success_count": sum(successes),
        "success_records": len(successes),
        "success_rate": sum(successes) / len(successes) if successes else None,
        "average_score": sum(scores) / len(scores) if scores else None,
        "score_records": len(scores),
        "metric_averages": averages,
        "metric_counts": counts,
    }


def report_text(summary):
    sections = []
    for item in [*summary["tasks"], {"task_name": "ALL EPISODES", **summary["overall"]}]:
        lines = [f"Task: {item['task_name']}", f"Records: {item['records']}"]
        if item["success_rate"] is not None:
            lines.append(f"Success rate: {item['success_count']}/{item['success_records']} ({item['success_rate']:.2%})")
        if item["average_score"] is not None:
            lines.append(f"Average score: {item['average_score']:.6f} (n={item['score_records']})")
        lines.append("Metric averages:")
        averages, counts = item["metric_averages"], item["metric_counts"]
        names = [name for name in METRIC_ORDER if name in averages]
        names += sorted(set(averages) - set(names))
        lines += [f"  {name}: {averages[name]:.6f} (n={counts[name]})" for name in names]
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def process_results(results_dir, repo_id=None, start_number=5, run_name=None, api=None, metadata=None):
    results_dir = Path(results_dir).expanduser().resolve()
    run_name = run_name or results_dir.name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("Run name must be a single folder name using letters, digits, '.', '_' or '-'")
    if start_number < 0:
        raise ValueError("start_number must be zero or greater")
    source = results_dir / "episode_results.jsonl"
    if not source.is_file():
        raise FileNotFoundError(
            f"RoboLab evaluation result not found: {source}\n"
            "No metrics were uploaded. Check the client evaluation log and its output directory. "
            "The result file is written after an episode finishes; verify --output-folder-name "
            "and the RoboLab checkout before rerunning post-processing."
        )
    groups = group_lines_by_task(source)
    tasks, all_records = [], []
    for number, (task, lines) in enumerate(groups.items(), start=start_number):
        records = [json.loads(line) for line in lines]
        tasks.append({"task_name": task, "file": f"RoboLab{number:02d}.jsonl", **summarize(records)})
        all_records.extend(records)
    summary = {"run_name": run_name, "repo_id": repo_id, "tasks": tasks, "overall": summarize(all_records)}
    report = report_text(summary)

    # Use a fresh staging folder so stale task files and large HDF5/video files
    # cannot accidentally enter the upload. Keep local artifacts even if HF fails.
    output = results_dir / "metrics"
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".metrics-", dir=results_dir) as temporary:
        staging = Path(temporary)
        shutil.copyfile(source, staging / source.name)
        for item, lines in zip(tasks, groups.values()):
            (staging / item["file"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
        (staging / "metrics_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "metrics_summary.txt").write_text(report, encoding="utf-8")
        if metadata is not None:
            (staging / "evaluation_config.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for artifact in staging.iterdir():
            shutil.copyfile(artifact, output / artifact.name)
        print(report, flush=True)
        if repo_id is None:
            print(f"Local metrics: {output}", flush=True)
            return summary, None
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        # An evaluation belongs to an existing model repository. A typo must
        # not create a new repository and silently separate results from weights.
        api.repo_info(repo_id=repo_id, repo_type="model")
        destination = f"evaluations/{run_name}"
        commit = api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(staging),
            path_in_repo=destination, commit_message=f"Add RoboLab metrics: {run_name}",
        )
    print(f"Uploaded metrics: https://huggingface.co/{repo_id}/tree/main/{destination}", flush=True)
    return summary, commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--hf-repo-id", help="Existing model repository: ACCOUNT/MODEL (omit for local metrics)")
    parser.add_argument("--start-number", type=int, default=5)
    parser.add_argument("--run-name", help="Evaluation folder name on Hugging Face (default: results directory name)")
    args = parser.parse_args()
    process_results(args.results_dir, args.hf_repo_id, args.start_number, args.run_name)


if __name__ == "__main__":
    main()
