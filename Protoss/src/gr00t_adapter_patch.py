"""Install the tracked adapter loader into the separately checked-out GR00T repo."""
from pathlib import Path
import subprocess


def ensure_adapter_inference(groot_root):
    root = Path(groot_root).resolve()
    directory = Path(__file__).resolve().parents[1] / "patches"
    patches = [directory / "gr00t_adapter_inference.patch",
               directory / "gr00t_adapter_inference_modes.patch"]
    command = ["git", "apply"]
    # The newest patch changes context from the first patch. Check it first.
    latest = subprocess.run(command + ["--reverse", "--check", str(patches[-1])],
                            cwd=root, capture_output=True, text=True)
    if latest.returncode == 0:
        return
    for patch in patches:
        applied = subprocess.run(command + ["--reverse", "--check", str(patch)],
                                 cwd=root, capture_output=True, text=True)
        if applied.returncode == 0:
            continue
        checked = subprocess.run(command + ["--check", str(patch)], cwd=root,
                                 capture_output=True, text=True)
        if checked.returncode:
            raise RuntimeError(f"GR00T adapter patch conflicts with the local checkout: {patch}. "
                               "Preserve its changes and review the patch.\n" + checked.stderr)
        subprocess.run(command + [str(patch)], cwd=root, check=True)
