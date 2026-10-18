"""Start the GR00T server with inference-time PEFT adapter loading enabled."""
from pathlib import Path
import argparse
import runpy
import sys


def main():
    if __package__:
        from .src.gr00t_adapter_patch import ensure_adapter_inference
    else:
        from src.gr00t_adapter_patch import ensure_adapter_inference
    if __package__:
        from .src.hf_model_sync import download_for_server
    else:
        from src.hf_model_sync import download_for_server
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hf-repo-id")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument("--hf-base-repo-id", default="nvidia/GR00T-N1.7-DROID")
    parser.add_argument("--hf-base-revision", default="main")
    args, remaining = parser.parse_known_args()
    if args.hf_repo_id:
        remaining = download_for_server(args, remaining)
    sys.argv = [sys.argv[0], *remaining]
    root = Path(__file__).resolve().parents[1] / "Issac-GR00T-N17"
    ensure_adapter_inference(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(root))
    runpy.run_module("gr00t.eval.run_gr00t_server", run_name="__main__")


if __name__ == "__main__":
    main()
