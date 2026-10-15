"""Install the tracked adapter loader into the separately checked-out GR00T repo."""
from pathlib import Path
import subprocess


def ensure_adapter_inference(groot_root):
    root = Path(groot_root).resolve()
    patch = Path(__file__).resolve().parents[1] / "patches/gr00t_adapter_inference.patch"
    command = ["git", "apply"]
    applied = subprocess.run(command + ["--reverse", "--check", str(patch)],
                             cwd=root, capture_output=True, text=True)
    if applied.returncode == 0:
        return
    checked = subprocess.run(command + ["--check", str(patch)], cwd=root,
                             capture_output=True, text=True)
    if checked.returncode:
        raise RuntimeError("GR00T adapter patch conflicts with the local checkout; "
                           "preserve its changes and review Protoss/patches/gr00t_adapter_inference.patch.\n"
                           + checked.stderr)
    subprocess.run(command + [str(patch)], cwd=root, check=True)
