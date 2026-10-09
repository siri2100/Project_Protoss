"""Download successful annotated DROID LeRobot v3 episodes for GR00T."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pyarrow.parquet as pq

CAMERAS = {"exterior_image_1_left": "observation.images.exterior_1_left",
           "wrist_image_left": "observation.images.wrist_left"}
REPO = "lerobot/droid_1.0.1"

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["download"])
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/source")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    if args.num_episodes < 3:
        parser.error("At least 3 episodes required")
    download_subset(args.dataset_dir, args.num_episodes, args.revision)

if __name__ == "__main__":
    main()
