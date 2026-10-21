"""Patch the optional actuator probe without depending on neighboring lines."""

import argparse
import ast
import os
from pathlib import Path
import shutil
import stat
import tempfile


SOURCE = Path("source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py")
OLD_PROBE = '_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None'
SAFE_PROBE = '''_HAS_NEWTON_ACTUATORS = (
    importlib.util.find_spec("isaaclab_newton") is not None
    and importlib.util.find_spec("isaaclab_newton.actuators") is not None
)'''


def probe_assignment(text):
    matches = [node for node in ast.parse(text).body if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "_HAS_NEWTON_ACTUATORS"
                       for target in node.targets)]
    if len(matches) != 1 or len(matches[0].targets) != 1:
        raise ValueError("Expected exactly one _HAS_NEWTON_ACTUATORS assignment; source left unchanged")
    return matches[0]


def ensure_optional_newton(root, check=False):
    source = Path(root) / SOURCE
    with source.open(encoding="utf-8", newline="") as stream:
        text = stream.read()
    assignment = probe_assignment(text)
    expression = ast.dump(assignment.value)
    if expression == ast.dump(probe_assignment(SAFE_PROBE).value):
        print(f"Optional Newton probe OK: {source}", flush=True)
        return False
    if expression != ast.dump(probe_assignment(OLD_PROBE).value):
        raise ValueError(f"Unrecognized Newton probe in {source}; source left unchanged")
    if check:
        raise ValueError("Optional Newton fix is not applied. Run setup successfully before smoke/eval.")

    lines = text.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in text else "\n"
    lines[assignment.lineno - 1:assignment.end_lineno] = [SAFE_PROBE.replace("\n", newline) + newline]
    updated = "".join(lines)
    ast.parse(updated)
    backup = source.with_suffix(".py.before_protoss_optional_newton")
    if not backup.exists():
        shutil.copy2(source, backup)
    # Preserve other source edits, newlines and permissions; replace atomically.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=source.parent,
                                         prefix=".protoss-newton-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
        temporary.chmod(stat.S_IMODE(source.stat().st_mode))
        os.replace(temporary, source)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f"Patched optional Newton probe: {source}; backup: {backup}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("isaaclab_root", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate without changing files")
    args = parser.parse_args()
    try:
        ensure_optional_newton(args.isaaclab_root, check=args.check)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
