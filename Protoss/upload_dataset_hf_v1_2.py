"""Back up generated A/B/test data to a private Hugging Face dataset repository."""
import argparse
import json
from pathlib import Path

from upload_checkpoint_hf_v1_2 import upload


def inventory(folder):
    def required(path):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Dataset is incomplete: missing or empty {path}")
        return path

    report = json.loads(required(folder / "styles.json").read_text())
    for split in ("a", "b", "test"):
        root = folder / split
        for name in ("info.json", "modality.json", "tasks.jsonl", "stats.json", "relative_stats.json"):
            required(root / "meta" / name)
        info = json.loads((root / "meta/info.json").read_text())
        episodes = [json.loads(line) for line in required(root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        if not episodes or len(episodes) != info["total_episodes"] or len(episodes) != len(report["splits"][split]):
            raise ValueError(f"Episode count mismatch in {split}")
        cameras = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
        if not cameras:
            raise ValueError(f"No video features in {split}")
        if sum(ep["length"] for ep in episodes) != info["total_frames"]:
            raise ValueError(f"Frame count mismatch in {split}")
        for ep in episodes:
            values = {"episode_index": ep["episode_index"],
                      "episode_chunk": ep["episode_index"] // info["chunks_size"]}
            paths = [info["data_path"].format(**values)]
            paths += [info["video_path"].format(**values, video_key=key) for key in cameras]
            for relative in paths:
                path = (root / relative).resolve()
                if root.resolve() not in path.parents:
                    raise ValueError("Dataset metadata path escapes dataset folder")
                required(path)
    return [p for p in folder.rglob("*") if p.is_file()
            and not any(part in (".git", ".cache", ".DS_Store") for part in p.relative_to(folder).parts)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True,
                        help="Generated groot_styles_retimed root containing a/, b/, test/, styles.json")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    upload(args, validator=inventory, repo_type="dataset")


if __name__ == "__main__":
    main()
