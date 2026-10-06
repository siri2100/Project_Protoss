"""Prepare retimed DROID styles and independently fine-tune GR00T A/B.

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


def retimed_splits(records, seed):
    """Split source IDs first; both training styles use the same sources."""
    if len(records) < 3:
        raise ValueError("At least 3 source episodes required")
    order = np.random.default_rng(seed).permutation(len(records))
    n_test = max(1, int(np.ceil(len(records) * .2)))
    test = [dict(records[i]) for i in order[:n_test]]
    train = [dict(records[i]) for i in order[n_test:]]
    return {"a": [dict(r) for r in train], "b": [dict(r) for r in train], "test": test}


def smooth_positions(values, window):
    """Centered moving average with a linear correction preserving endpoints."""
    if window == 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack([np.convolve(padded[:, i], np.ones(window) / window, mode="valid")
                         for i in range(values.shape[1])], axis=1)
    fraction = np.linspace(0, 1, len(values))[:, None]
    return smoothed + (1 - fraction) * (values[0] - smoothed[0]) + fraction * (values[-1] - smoothed[-1])


def retime_frame(frame, fps, speed_factor, smooth_window=1):
    """Return synthesized 15Hz lowdim data and shared source indices for RGB.

    Images and gripper use nearest samples; continuous state/action interpolate.
    Only action joint/EEF targets are smoothed, retaining observed state.
    """
    from scipy.spatial.transform import Rotation, Slerp
    if not np.isfinite(speed_factor) or speed_factor <= 0 or smooth_window < 1 or smooth_window % 2 != 1:
        raise ValueError("Speed must be positive and smoothing window a positive odd integer")
    count = len(frame)
    if count < 4 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("At least 4 source frames and a positive fps are required")
    # linspace includes both endpoints and avoids an artificial short last step.
    output_count = max(4, int(round((count - 1) / speed_factor)) + 1)
    timeline = np.linspace(0, count - 1, output_count)
    nearest = np.floor(timeline + .5).astype(int)
    # Keep only supported low-dimensional fields; do not copy stale velocities.
    result = frame.iloc[nearest][["language_instruction"]].copy().reset_index(drop=True)
    for prefix in ("observation.state", "action"):
        joints = vector_column(frame, prefix + ".joint_position", 7)
        cart = vector_column(frame, prefix + ".cartesian_position", 6)
        rotations = Rotation.from_euler("XYZ", cart[:, 3:])
        if prefix == "action" and smooth_window > 1:
            joints = smooth_positions(joints, smooth_window)
            cart[:, :3] = smooth_positions(cart[:, :3], smooth_window)
            pad = smooth_window // 2
            filtered = [rotations[max(0, i - pad):min(count, i + pad + 1)].mean().as_quat()
                        for i in range(count)]
            filtered[0], filtered[-1] = rotations[0].as_quat(), rotations[-1].as_quat()
            rotations = Rotation.from_quat(filtered)
        def interpolate(values):
            return np.stack([np.interp(timeline, np.arange(count), values[:, i])
                             for i in range(values.shape[1])], axis=1).astype(np.float32)
        result[prefix + ".joint_position"] = list(interpolate(joints))
        xyz = interpolate(cart[:, :3])
        euler = Slerp(np.arange(count), rotations)(timeline).as_euler("XYZ")
        result[prefix + ".cartesian_position"] = list(np.concatenate([xyz, euler], axis=-1).astype(np.float32))
        grip = vector_column(frame, prefix + ".gripper_position", 1)
        result[prefix + ".gripper_position"] = list(grip[nearest])
    result["frame_index"] = np.arange(output_count)
    result["timestamp"] = np.arange(output_count) / fps
    return result, nearest


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


def select_styles(records, seed, allow_empty=False):
    """Reserve shared test episodes before ranking; balance task counts A/B."""
    rng = np.random.default_rng(seed)
    groups = {}
    for record in records:
        groups.setdefault(record["task"], []).append(record)
    splits = {"a": [], "b": [], "test": []}
    skipped = []
    for task, group in sorted(groups.items()):
        if len(group) < 5:
            skipped.append({"task": task, "episodes": len(group), "reason": "fewer than 5 episodes"})
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
            skipped.append({"task": task, "episodes": len(group),
                            "reason": "no clear speed/acceleration contrast",
                            "candidate_means": {name: {k: float(np.mean([r[k] for r in subset]))
                                for k in ("speed_rms_rad_s", "accel_rms_rad_s2")}
                                for name, subset in (("a", a), ("b", b))}})
            continue
        splits["a"].extend(a)
        splits["b"].extend(b)
        splits["test"].extend(test)
    if not splits["a"] and not allow_empty:
        raise ValueError("No task has >=5 episodes and a slow/smooth vs fast contrast. "
                         "Download more successful episodes or collect paired styles; "
                         "a general-purpose DROID subset may be insufficient.")
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
    method = getattr(args, "style_method", "retime")
    if method == "retime":
        splits, skipped = retimed_splits(records, args.seed), []
    else:
        splits, skipped = select_styles(records, args.seed, allow_empty=True)
    counts = {}
    for record in records:
        counts[record["task"]] = counts.get(record["task"], 0) + 1
    inspection_path = output.parent / (output.name + ".inspection.json")
    write_json(inspection_path, {"source": str(source), "revision": manifest["revision"],
        "seed": args.seed, "episodes": len(records), "unique_tasks": len(counts),
        "task_counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
        "tasks_with_at_least_5_episodes": sum(n >= 5 for n in counts.values()),
        "selected_counts": {k: len(v) for k, v in splits.items()},
        "skipped_tasks": skipped, "episode_metrics": records})
    print(f"Inspection: {len(records)} episodes, {len(counts)} unique instructions, "
          f"{sum(n >= 5 for n in counts.values())} tasks with >=5 episodes", flush=True)
    for task, count in sorted(counts.items(), key=lambda item: -item[1])[:10]:
        print(f"  {count:4d} episodes: {task}", flush=True)
    print(f"Saved inspection: {inspection_path}", flush=True)
    print(f"Style method: {method}; selected A={len(splits['a'])}, "
          f"B={len(splits['b'])}, test={len(splits['test'])}", flush=True)
    if getattr(args, "inspect_only", False):
        return
    if not splits["a"]:
        raise ValueError(f"No usable style groups. See {inspection_path} for task counts "
                         "and exclusion reasons. Collect repeated successful demonstrations "
                         "of the same task with distinct motion styles.")
    from gr00t.data.state_action.droid_frame import compute_eef_9d
    from gr00t.data.stats import generate_stats, generate_rel_stats
    from gr00t.data.embodiment_tags import EmbodimentTag
    rows_by_id = {int(r["episode_index"]): r for r in manifest["episodes"]}
    tasks = sorted({r["task"] for split in splits.values() for r in split})
    # Use one time map for both cameras and all state/action fields.
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
            source_count = len(frame)
            nearest = np.arange(source_count)
            if method == "retime" and name in ("a", "b"):
                factor = args.slow_factor if name == "a" else args.fast_factor
                window = args.smooth_window if name == "a" else 1
                record["source_metrics"] = {key: record[key] for key in
                    ("speed_rms_rad_s", "accel_rms_rad_s2", "jerk_rms_rad_s3", "duration_s")}
                frame, nearest = retime_frame(frame, fps, factor, window)
                record["speed_factor"] = factor
                record["smooth_window"] = window
                record.update(motion_metrics(vector_column(frame, "action.joint_position", 7), fps))
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
                if abs(duration * fps - source_count) > .1:
                    raise ValueError("Video duration does not match episode frames")
                rgb = decode_episode(source / video_filename(row, key), start, source_count, fps)[nearest]
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
               "style_method": method,
               "synthetic": method == "retime",
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


def main(argv=None, fixed_style=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-root", type=Path, required=True)
    if fixed_style is None:
        sub = parser.add_subparsers(dest="command", required=True)
        prep = sub.add_parser("prepare")
        prep.add_argument("--dataset-dir", type=Path, required=True)
        prep.add_argument("--output-dir", type=Path, required=True)
        prep.add_argument("--seed", type=int, default=42)
        prep.add_argument("--style-method", choices=["retime", "select"], default="retime")
        prep.add_argument("--slow-factor", type=float, default=.75)
        prep.add_argument("--fast-factor", type=float, default=1.25)
        prep.add_argument("--smooth-window", type=int, default=5)
        prep.add_argument("--inspect-only", action="store_true",
                          help="Write task counts, exclusion reasons and motion metrics without converting videos")
        training = sub.add_parser("train")
    else:
        if fixed_style not in ("a", "b"):
            raise ValueError("fixed_style must be a or b")
        parser.description = f"Fine-tune GR00T model {fixed_style.upper()}"
        parser.set_defaults(command="train", style=fixed_style)
        training = parser
    training.add_argument("--styles-dir", type=Path, required=True)
    if fixed_style is None:
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
    args = parser.parse_args(argv)
    if args.command == "prepare" and (not 0 < args.slow_factor < 1 or not np.isfinite(args.fast_factor)
            or args.fast_factor <= 1 or args.smooth_window < 1 or args.smooth_window % 2 != 1):
        parser.error("slow-factor must be (0,1), fast-factor >1, smooth-window positive and odd")
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
