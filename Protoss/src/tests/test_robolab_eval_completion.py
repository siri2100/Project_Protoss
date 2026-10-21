"""Native exit(0) must not turn a partially evaluated run into a complete one."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CompletionTests(unittest.TestCase):
    def run_fixture(self, fail_evaluation=False, native_close=False):
        patch = Path(__file__).resolve().parents[2] / "patches/robolab_eval_completion.patch"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "policies/gr00t/run.py"
            target.parent.mkdir(parents=True)
            target.write_text('''import os
from types import SimpleNamespace
args_cli = None
POLICY = "gr00t"
make_client = None
def run_evaluation(*args, **kwargs):
    print("partial episode results exist", flush=True)
    if FAIL_EVALUATION: os._exit(0)
def close():
    if NATIVE_CLOSE: os._exit(0)
simulation_app = SimpleNamespace(close=close)
def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


main()
'''.replace("FAIL_EVALUATION", repr(fail_evaluation)).replace("NATIVE_CLOSE", repr(native_close)))
            subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                           cwd=root, check=True, capture_output=True)
            return subprocess.run([sys.executable, str(target)], capture_output=True, text=True)

    def test_normal_completion_is_marked(self):
        result = self.run_fixture()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROTOSS ROBOLAB EVALUATION COMPLETE", result.stdout)

    def test_native_cleanup_exit_after_completed_evaluation_is_marked(self):
        result = self.run_fixture(native_close=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PROTOSS ROBOLAB EVALUATION COMPLETE", result.stdout)

    def test_native_exit_during_evaluation_is_not_marked_despite_zero_exit(self):
        result = self.run_fixture(fail_evaluation=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("partial episode results exist", result.stdout)
        self.assertNotIn("PROTOSS ROBOLAB EVALUATION COMPLETE", result.stdout)


if __name__ == "__main__":
    unittest.main()
