"""Prepare task-matched DROID styles and independently fine-tune GR00T A/B.

Run with the GR00T N1.7 Python environment, not .venv-v12.
Style is learned from demonstrations using the unmodified GR00T flow loss.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from prepare_droid_v1_2 import (
    CAMERAS, data_filename, video_filename, vector_column, decode_episode,
)

TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


def motion_metrics(joints, fps):
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 2 or joints.shape[1] != 7 or len(joints) < 4:
        raise ValueError("Expected at least four frames of seven joint commands")
    if fps <= 0 or not np.isfinite(joints).all():
        raise ValueError("Nonfinite commands or invalid fps")
    velocity = np.diff(joints, axis=0) * fps
    acceleration = np.diff(velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    return {"speed_rms_rad_s": float(np.sqrt(np.mean(velocity ** 2))),
            "accel_rms_rad_s2": float(np.sqrt(np.mean(acceleration ** 2))),
            "jerk_rms_rad_s3": float(np.sqrt(np.mean(jerk ** 2))),
            "duration_s": len(joints) / fps}


def select_styles(records, seed):
    """Reserve shared test episodes before ranking; balance task counts A/B."""
    rng = np.random.default_rng(seed)
    groups = {}
    for record in records:
        groups.setdefault(record["task"], []).append(record)
    splits = {"a": [], "b": [], "test": []}
    skipped = []
    for task, group in sorted(groups.items()):
        if len(group) < 5:
            skipped.append({"task": task, "episodes": len(group)})
            continue
        group = [group[i] for i in rng.permutation(len(group))]
        test_count = max(1, int(len(group) * .2))
        test, train = group[:test_count], group[test_count:]
        # Candidate A: low speed AND low acceleration within the same task.
        def ranks(key):
            order = np.argsort([r[key] for r in train], kind="stable")
            rank = np.empty(len(train), dtype=float)
            rank[order] = np.arange(len(train))
            return rank
        score = ranks("speed_rms_rad_s") + ranks("accel_rms_rad_s2")
        n = len(train) // 2
        a_indices = set(np.argsort(score, kind="stable")[:n].tolist())
        a = [r for i, r in enumerate(train) if i in a_indices]
        b = sorted([r for i, r in enumerate(train) if i not in a_indices],
                   key=lambda r: r["speed_rms_rad_s"], reverse=True)[:n]
        # Fail instead of naming two indistinguishable subsets smooth/fast.
        if any(np.mean([r[k] for r in a]) >= np.mean([r[k] for r in b])
               for k in ("speed_rms_rad_s", "accel_rms_rad_s2")):
            skipped.append({"task": task, "reason": "no clear speed/acceleration contrast"})
            continue
        splits["a"].extend(a)
        splits["b"].extend(b)
        splits["test"].extend(test)
    if not splits["a"]:
        raise ValueError("No task has >=5 episodes and a slow/smooth vs fast contrast. "
                         "Download more successful episodes or collect paired styles; "
                         "the existing 30-episode refiner subset may be insufficient.")
    return splits, skipped


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def prepare(args):
    source = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is nonempty; use a new directory")
    import pyarrow.parquet as pq
    from gr00t.data.state_action.droid_frame import compute_eef_9d
    from gr00t.data.stats import generate_stats, generate_rel_stats
    from gr00t.data.embodiment_tags import EmbodimentTag
    manifest = json.loads((source / "subset.json").read_text())
    fps = float(manifest["fps"])
    records = []
    shard_cache = None
    last_path = None
    for row in manifest["episodes"]:
        path = source / data_filename(row)
        if path != last_path:
            shard_cache = pq.read_table(path).to_pandas()
            last_path = path
        ep = int(row["episode_index"])
        frame = shard_cache[shard_cache["episode_index"] == ep].sort_values("frame_index")
        count = int(row["length"])
        if len(frame) != count or not np.array_equal(frame["frame_index"], np.arange(count)):
            raise ValueError(f"Incomplete episode {ep}")
        if not np.allclose(frame["timestamp"], np.arange(count) / fps, atol=1e-3):
            raise ValueError(f"Invalid timestamps in episode {ep}")
        tasks = frame["language_instruction"].astype(str).str.strip().unique().tolist()
        if len(tasks) != 1 or not tasks[0]:
            raise ValueError(f"Episode {ep} must have one nonempty instruction")
        records.append({"source_episode": ep, "task": tasks[0],
                        **motion_metrics(vector_column(frame, "action.joint_position", 7), fps)})
    splits, skipped = select_styles(records, args.seed)
    rows_by_id = {int(r["episode_index"]): r for r in manifest["episodes"]}
    tasks = sorted({r["task"] for split in splits.values() for r in split})
    # Materialize genuine, synchronized episodes; no action-only retiming.
    for name, selected in splits.items():
        root = output / name
        meta = root / "meta"
        meta.mkdir(parents=True)
        episodes = []
        global_index = 0
        last_path = None
        for index, record in enumerate(selected):
            row = rows_by_id[record["source_episode"]]
            path = source / data_filename(row)
            if path != last_path:
                shard_cache = pq.read_table(path).to_pandas()
                last_path = path
            frame = shard_cache[shard_cache["episode_index"] == record["source_episode"]].sort_values("frame_index").copy()
            count = len(frame)
            for prefix in ("observation.state", "action"):
                eef = compute_eef_9d(vector_column(frame, prefix + ".cartesian_position", 6))
                grip = vector_column(frame, prefix + ".gripper_position", 1)
                joints = vector_column(frame, prefix + ".joint_position", 7)
                if ((grip < 0) | (grip > 1)).any():
                    raise ValueError("Expected DROID gripper commands in [0,1]")
                frame[prefix] = list(np.concatenate([eef, grip, joints], axis=-1).astype(np.float32))
            frame["episode_index"] = index
            frame["index"] = np.arange(global_index, global_index + count)
            frame["task_index"] = tasks.index(record["task"])
            global_index += count
            data_path = root / f"data/chunk-{index // 1000:03d}/episode_{index:06d}.parquet"
            data_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(data_path, index=False)
            for key in CAMERAS.values():
                start = float(row[f"videos/{key}/from_timestamp"])
                duration = float(row[f"videos/{key}/to_timestamp"]) - start
                if abs(duration * fps - count) > .1:
                    raise ValueError("Video duration does not match episode frames")
                rgb = decode_episode(source / video_filename(row, key), start, count, fps)
                target = root / f"videos/chunk-{index // 1000:03d}/{key}/episode_{index:06d}.mp4"
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", "320x180", "-r", str(fps), "-i", "pipe:0", "-an",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "1", str(target)],
                               input=rgb.tobytes(), check=True)
            episodes.append({"episode_index": index, "tasks": [record["task"]], "length": count})
        write_jsonl(meta / "episodes.jsonl", episodes)
        write_jsonl(meta / "tasks.jsonl", [{"task_index": i, "task": task} for i, task in enumerate(tasks)])
        features = {key: {"dtype": "video", "shape": [180, 320, 3]} for key in CAMERAS.values()}
        features.update({"observation.state": {"dtype": "float32", "shape": [17]},
                         "action": {"dtype": "float32", "shape": [17]},
                         "task_index": {"dtype": "int64", "shape": [1]}})
        write_json(meta / "info.json", {"codebase_version": "v2.1", "robot_type": "droid", "fps": fps,
                   "total_episodes": len(episodes), "total_frames": global_index, "chunks_size": 1000,
                   "total_chunks": (len(episodes) + 999) // 1000, "total_tasks": len(tasks),
                   "total_videos": len(episodes) * 2, "splits": {"train": f"0:{len(episodes)}"},
                   "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                   "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                   "features": features})
        layout = {"eef_9d": {"start": 0, "end": 9}, "gripper_position": {"start": 9, "end": 10},
                  "joint_position": {"start": 10, "end": 17}}
        write_json(meta / "modality.json", {"state": layout, "action": layout,
                   "video": {key.split(".")[-1]: {"original_key": key} for key in CAMERAS.values()},
                   "annotation": {"language.language_instruction": {"original_key": "task_index"}}})
        generate_stats(root)
        generate_rel_stats(root, EmbodimentTag.resolve(TAG))
    write_json(output / "styles.json", {"source": str(source), "revision": manifest["revision"],
               "seed": args.seed, "fps": fps, "splits": splits, "skipped_tasks": skipped,
               "metrics": "finite differences of absolute joint COMMANDS, not measured robot motion"})
    print(f"Prepared A={len(splits['a'])}, B={len(splits['b'])}, test={len(splits['test'])}: {output}")


def train(args):
    import torch
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run

    dataset = args.styles_dir.resolve() / args.style
    report = json.loads((args.styles_dir / "styles.json").read_text())
    if not report["splits"][args.style] or not (dataset / "meta/relative_stats.json").exists():
        raise ValueError("Run prepare first; style dataset/statistics are missing")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("Output is nonempty; choose a new path or --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("GR00T fine-tuning requires a CUDA GPU")
    config = get_default_config().load_dict({"data": {"download_cache": False, "datasets": [
        {"dataset_paths": [str(dataset)], "mix_ratio": 1.0,
         "embodiment_tag": EmbodimentTag.resolve(TAG).value}]}})
    config.load_config_path = None
    config.data.seed = args.seed
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.training.start_from_checkpoint = str(args.base_model.resolve())
    config.training.output_dir = str(args.output_dir.resolve())
    # Upstream appends experiment_name to output_dir when it is non-None.
    config.training.experiment_name = None
    config.training.num_gpus = 1
    config.training.global_batch_size = args.batch_size
    config.training.gradient_accumulation_steps = args.gradient_accumulation
    config.training.learning_rate = args.lr
    config.training.max_steps = args.max_steps
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = 3
    config.training.dataloader_num_workers = args.workers
    config.training.optim = "adamw_torch"
    config.training.use_wandb = False
    config.training.resume_from_checkpoint = args.resume
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "style_run.json", {"style": args.style, "base_model": str(args.base_model.resolve()),
               "dataset": str(dataset), "seed": args.seed, "loss": "native GR00T masked flow-matching MSE",
               "selection": report, "batch_size": args.batch_size, "lr": args.lr,
               "max_steps": args.max_steps, "gradient_accumulation": args.gradient_accumulation})
    run(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--dataset-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=42)
    training = sub.add_parser("train")
    training.add_argument("--styles-dir", type=Path, required=True)
    training.add_argument("--style", choices=["a", "b"], required=True)
    training.add_argument("--base-model", type=Path, required=True)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--max-steps", type=int, default=2000)
    training.add_argument("--save-steps", type=int, default=500)
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--gradient-accumulation", type=int, default=16)
    training.add_argument("--lr", type=float, default=1e-5)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "train" and (min(args.max_steps, args.save_steps, args.batch_size,
                                        args.gradient_accumulation) < 1 or args.lr <= 0 or args.workers < 0):
        parser.error("Steps/batch/accumulation/lr must be positive; workers >= 0")
    sys.path.insert(0, str(args.groot_root.resolve()))
    if args.command == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
