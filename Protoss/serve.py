"""Start the GR00T server with inference-time PEFT adapter loading enabled."""
from pathlib import Path
import runpy
import sys


def main():
    if __package__:
        from .src.gr00t_adapter_patch import ensure_adapter_inference
    else:
        from src.gr00t_adapter_patch import ensure_adapter_inference
    root = Path(__file__).resolve().parents[1] / "Issac-GR00T-N17"
    ensure_adapter_inference(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(root))
    runpy.run_module("gr00t.eval.run_gr00t_server", run_name="__main__")


if __name__ == "__main__":
    main()
