import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class DebugEntrypointTests(unittest.TestCase):
    def run_entrypoint(self, setup_exit=0, mode="check", extra=()):
        script = Path(__file__).resolve().parents[2] / "debug_robolab_595.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            debug_root = root / "debug checkout"
            runner = debug_root / "Protoss/setup_robolab_595.sh"
            runner.parent.mkdir(parents=True)
            runner.write_text('''#!/usr/bin/env bash
printf '%s\n' "$1" >> "$DEBUG_TEST_LOG"
printf '%s\n' "$PROTOSS_ROOT" >> "$DEBUG_TEST_LOG"
if [ "$1" = setup ]; then exit "$DEBUG_SETUP_EXIT"; fi
printf '%s\n' "$@" >> "$DEBUG_TEST_LOG"
''')
            log = root / "commands"
            env = {**os.environ, "PROTOSS_DEBUG_ROOT": str(debug_root),
                   "PROTOSS_ROOT": "/workspace/Project_Protoss",  # Must not select the training checkout.
                   "DEBUG_TEST_LOG": str(log), "DEBUG_SETUP_EXIT": str(setup_exit)}
            result = subprocess.run(["bash", str(script), mode, *extra], env=env,
                                    text=True, capture_output=True)
            return result, log.read_text().splitlines(), str(debug_root)

    def test_check_stops_after_failed_setup(self):
        result, lines, root = self.run_entrypoint(setup_exit=9)
        self.assertEqual(result.returncode, 9)
        self.assertEqual(lines, ["setup", root])

    def test_successful_setup_then_smoke_use_debug_checkout(self):
        result, lines, root = self.run_entrypoint(extra=("--task", "BananaOnPlateTask"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, ["setup", root, "smoke", root, "smoke", "--task", "BananaOnPlateTask"])

    def test_eval_arguments_are_forwarded(self):
        result, lines, root = self.run_entrypoint(mode="eval", extra=("--output-folder-name", "debug-run"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(lines, ["eval", root, "eval", "--output-folder-name", "debug-run"])


if __name__ == "__main__":
    unittest.main()
