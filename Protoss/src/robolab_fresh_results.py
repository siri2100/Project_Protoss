"""Start each evaluation with empty results while preserving previous runs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import wraps
import json
import os
from pathlib import Path
from uuid import uuid4


@contextmanager
def fresh_output_directory(root, name):
    """Archive an old run under _previous; hold a process lock through evaluation."""
    if not name or name in (".", "..", "_previous", ".evaluation-locks") or Path(name).name != name:
        raise ValueError("Evaluation output name must be a single folder name")
    root = Path(root).resolve()
    locks = root / ".evaluation-locks"
    locks.mkdir(parents=True, exist_ok=True)
    output = root / name
    # Keep lock files: unlinking would let another process lock a different inode.
    # The OS releases flock even if Kit exits natively without Python cleanup.
    with (locks / f"{name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Evaluation already running in {output}; choose another result name") from None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{stamp}-{uuid4().hex[:12]}"
        previous = None
        if output.is_symlink() or (output.exists() and not output.is_dir()):
            raise ValueError(f"Evaluation output must be a directory, not a file/symlink: {output}")
        if output.exists():
            previous = root / "_previous" / name / run_id
            previous.parent.mkdir(parents=True, exist_ok=True)
            output.rename(previous)
            print(f"Archived previous evaluation: {previous}", flush=True)
        output.mkdir()
        (output / "evaluation_run.json").write_text(json.dumps({
            "run_id": run_id, "pid": os.getpid(), "output_dir": str(output),
            "previous_output": str(previous) if previous else None,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Fresh evaluation output: {output}", flush=True)
        try:
            yield output
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def fresh_evaluation(evaluate):
    """Wrap RoboLab's run_evaluation before it reads JSONL or opens HDF5."""
    @wraps(evaluate)
    def run(args, *positional, **kwargs):
        from robolab.constants import PACKAGE_DIR

        if args.output_folder_name is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            args.output_folder_name = f"{stamp}_{kwargs.get('policy', 'gr00t')}"
        with fresh_output_directory(Path(PACKAGE_DIR) / "output", args.output_folder_name):
            return evaluate(args, *positional, **kwargs)
    return run
