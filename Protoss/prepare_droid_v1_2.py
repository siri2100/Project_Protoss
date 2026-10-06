"""Download a small public DROID LeRobot v3 subset and export flat GR00T NPZs.

No LeRobot installation, manual camera mapping, or raw-data conversion required.
The pinned GR00T source supplies its own DROID EEF conversion function.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq

CAMERAS = {
    "exterior_image_1_left": "observation.images.exterior_1_left",
    "wrist_image_left": "observation.images.wrist_left",
}
REPO = "lerobot/droid_1.0.1"
GROOT_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_episodes(metadata_path, count):
    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
               "stats/is_episode_successful/min"]
    for key in CAMERAS.values():
        columns += [f"videos/{key}/{field}" for field in
                    ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    rows = pq.read_table(metadata_path, columns=columns).to_pylist()
    selected = []
    for row in rows:
        tasks = [str(t).strip() for t in row["tasks"] if str(t).strip()]
        if not np.asarray(row["stats/is_episode_successful/min"]).all() or not tasks:
            continue
        row.pop("stats/is_episode_successful/min")
        row["tasks"] = tasks
        selected.append(row)
        if len(selected) == count:
            return selected
    raise ValueError(f"First metadata shard contains only {len(selected)} successful annotated episodes; requested {count}")


def data_filename(row):
    return f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"


def video_filename(row, key):
    return (f"videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/"
            f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")


def download_subset(directory, count, revision):
    # Use standard Hub downloads without the optional hf_transfer package.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    from huggingface_hub import HfApi, hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "subset.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if len(existing["episodes"]) != count:
            raise ValueError("Existing subset has a different episode count; use a new dataset directory")
        if revision != "main" and revision != existing["revision"]:
            raise ValueError("Existing subset revision differs; use a new dataset directory")
        resolved = existing["revision"]
    else:
        resolved = HfApi().dataset_info(REPO, revision=revision).sha

    def fetch(name):
        return Path(hf_hub_download(REPO, name, repo_type="dataset", revision=resolved,
                                    local_dir=directory))

    info = json.loads(fetch("meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or float(info["fps"]) != 15:
        raise ValueError("Expected DROID LeRobot v3.0 at 15 Hz")
    rows = select_episodes(fetch("meta/episodes/chunk-000/file-000.parquet"), count)
    files = {data_filename(row) for row in rows}
    files.update(video_filename(row, key) for row in rows for key in CAMERAS.values())
    for name in sorted(files):
        print(f"Downloading {name}", flush=True)
        fetch(name)
    manifest = {"repo_id": REPO, "revision": resolved, "fps": 15, "episodes": rows,
                "selection": "first successful episodes with nonempty language in metadata shard 000"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Subset ready: {count} episodes, {len(files)} data/video shards, revision={resolved}")


def split_episodes(rows, seed):
    if len(rows) < 3:
        raise ValueError("At least 3 episodes required for train/val/test")
    indices = np.random.default_rng(seed).permutation(len(rows))
    n_val = max(1, int(round(len(rows) * .1)))
    n_test = max(1, int(round(len(rows) * .1)))
    n_train = len(rows) - n_val - n_test
    return {"train": [rows[i] for i in indices[:n_train]],
            "val": [rows[i] for i in indices[n_train:n_train + n_val]],
            "test": [rows[i] for i in indices[n_train + n_val:]]}


def vector_column(frame, key, dim):
    result = np.stack([np.asarray(v, dtype=np.float32).reshape(-1) for v in frame[key]])
    if result.shape != (len(frame), dim) or not np.isfinite(result).all():
        raise ValueError(f"Invalid {key}: expected finite [{len(frame)},{dim}]")
    return result


def decode_episode(path, start_seconds, count, fps):
    # Accurate input seek plus decode/re-encode-free RGB output: no stream-copy
    # cuts on keyframes. FFmpeg's AV1 support is used instead of OpenCV's backend.
    command = ["ffmpeg", "-v", "error", "-ss", f"{start_seconds:.9f}", "-i", str(path),
               "-an", "-vf", "scale=320:180", "-frames:v", str(count),
               "-threads", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"FFmpeg failed for {path}: {result.stderr.decode(errors='replace')}")
    expected = count * 180 * 320 * 3
    if len(result.stdout) != expected:
        raise ValueError(f"Video {path} has {len(result.stdout) // (180 * 320 * 3)} decoded frames; expected {count}")
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(count, 180, 320, 3)


def validate_config(config, horizon):
    if set(config["video"]["modality_keys"]) != set(CAMERAS):
        raise ValueError("Server must use the two DROID cameras")
    if not set(config["state"]["modality_keys"]) <= {"joint_position", "gripper_position", "eef_9d"}:
        raise ValueError("Unsupported state modalities")
    for group in ("video", "state"):
        offsets = config[group]["delta_indices"]
        if not offsets or 0 not in offsets or any(not isinstance(x, int) or x > 0 for x in offsets):
            raise ValueError("Exporter supports current/past observation offsets including zero")
    if config["action"]["delta_indices"][:horizon] != list(range(horizon)):
        raise ValueError("Action horizon must fit consecutive server offsets starting at zero")


def export_subset(dataset_dir, output_dir, config, eef_converter, horizon=8, stride=8, seed=42):
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Export directory is nonempty; use a new directory to preserve split integrity")
    validate_config(config, horizon)
    manifest = json.loads((dataset_dir / "subset.json").read_text())
    splits = split_episodes(manifest["episodes"], seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"dataset": manifest["repo_id"], "revision": manifest["revision"], "seed": seed,
              "horizon": horizon, "stride": stride, "fps": manifest["fps"],
              "modality_config": config, "splits": {}}
    # Read only required columns and one data shard at a time.
    columns = ["episode_index", "frame_index", "timestamp", "language_instruction",
               "observation.state.joint_position", "observation.state.gripper_position",
               "observation.state.cartesian_position", "action.joint_position", "action.gripper_position"]
    last_path, shard = None, None
    for split, rows in splits.items():
        split_report = []
        (output_dir / split).mkdir(exist_ok=True)
        for row in rows:
            path = dataset_dir / data_filename(row)
            if path != last_path:
                shard = pq.read_table(path, columns=columns).to_pandas()
                last_path = path
            ep = int(row["episode_index"])
            frame = shard[shard["episode_index"] == ep].sort_values("frame_index")
            count = int(row["length"])
            if len(frame) != count or not np.array_equal(frame["frame_index"].to_numpy(), np.arange(count)):
                raise ValueError(f"Episode {ep} is incomplete in data shard; cannot align actions/video")
            timestamps = frame["timestamp"].to_numpy(dtype=float)
            if not np.allclose(timestamps, np.arange(count) / manifest["fps"], atol=1e-3):
                raise ValueError(f"Episode {ep} timestamps do not follow 15 Hz frame indices")
            state = {"joint_position": vector_column(frame, "observation.state.joint_position", 7),
                     "gripper_position": vector_column(frame, "observation.state.gripper_position", 1),
                     "eef_9d": np.asarray(eef_converter(vector_column(frame, "observation.state.cartesian_position", 6)), dtype=np.float32)}
            target = np.concatenate([vector_column(frame, "action.joint_position", 7),
                                     vector_column(frame, "action.gripper_position", 1)], axis=-1)
            if ((target[:, 7] < 0) | (target[:, 7] > 1)).any():
                raise ValueError("DROID gripper command must use 0=open, 1=closed")
            videos = {}
            for name, key in CAMERAS.items():
                start = float(row[f"videos/{key}/from_timestamp"])
                duration = float(row[f"videos/{key}/to_timestamp"]) - start
                if abs(duration * manifest["fps"] - count) > .1:
                    raise ValueError(f"Episode {ep} video timestamps disagree with data length")
                videos[name] = decode_episode(dataset_dir / video_filename(row, key), start, count, manifest["fps"])
            # Do not duplicate nonexistent history frames at episode start.
            first = max(-min(config[g]["delta_indices"]) for g in ("video", "state"))
            sample_count = 0
            for t in range(first, count, stride):
                prompt = str(frame["language_instruction"].iloc[t]).strip()
                if not prompt:
                    raise ValueError(f"Episode {ep} has empty language at frame {t}")
                obs = {}
                for group, values in (("video", videos), ("state", state)):
                    indices = t + np.array(config[group]["delta_indices"])
                    for name in config[group]["modality_keys"]:
                        obs[f"{group}.{name}"] = values[name][indices][None]
                for name in config["language"]["modality_keys"]:
                    obs[name] = np.array([prompt])
                valid = min(horizon, count - t)
                chunk = np.repeat(target[-1:], horizon, axis=0)
                chunk[:valid] = target[t:t + valid]
                mask = np.zeros((1, horizon), dtype=np.float32)
                mask[0, :valid] = 1
                dest = output_dir / split / f"episode_{ep:06d}" / f"frame_{t:06d}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, **obs,
                    **{"target.joint_position": chunk[None, :, :7],
                       "target.gripper_position": chunk[None, :, 7:]},
                    mask=mask, episode_id=np.array([f"{manifest['repo_id']}:{ep}"]))
                sample_count += 1
            if not sample_count:
                raise ValueError(f"Episode {ep} is too short for required observation history")
            split_report.append({"episode": ep, "frames": count, "samples": sample_count, "tasks": row["tasks"]})
            print(f"{split}: episode={ep}, samples={sample_count}", flush=True)
        report["splits"][split] = split_report
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Export ready: {output_dir / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download", "export"])
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/v1.2/droid"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/v1.2/export"))
    parser.add_argument("--num-episodes", type=int, default=30)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n17-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--modality-config", type=Path, help="Saved server config JSON instead of live server")
    parser.add_argument("--groot-root", type=Path, default=Path(__file__).resolve().parents[1] / "Issac-GR00T-N17")
    args = parser.parse_args()
    if args.num_episodes < 3 or args.horizon <= 0 or args.stride <= 0:
        parser.error("Need at least 3 episodes and positive horizon/stride")
    if args.mode == "download":
        download_subset(args.dataset_dir, args.num_episodes, args.revision)
        return
    revision = subprocess.check_output(["git", "-C", str(args.groot_root), "rev-parse", "HEAD"], text=True).strip()
    if revision != GROOT_REVISION:
        parser.error(f"GR00T source must be {GROOT_REVISION}")
    frame_module = load_module("droid_frame", args.groot_root / "gr00t/data/state_action/droid_frame.py")
    if args.modality_config:
        config = json.loads(args.modality_config.read_text())
    else:
        transport = load_module("transport", Path(__file__).with_name("main_v1.0.py"))
        client = transport.ModelClient(args.n17_endpoint)
        try:
            config = client.call("get_modality_config")
        finally:
            client.close()
    export_subset(args.dataset_dir, args.output_dir, config, frame_module.compute_eef_9d,
                  args.horizon, args.stride, args.seed)


if __name__ == "__main__":
    main()
