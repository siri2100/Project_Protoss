"""Exercise deployment against unpatched source files, without a GPU install."""
import ast
from pathlib import Path
import subprocess
import tempfile
import unittest

from src.gr00t_adapter_patch import ensure_adapter_inference


class AdapterPatchTests(unittest.TestCase):
    def test_patch_applies_and_second_application_is_noop(self):
        upstream = Path(__file__).resolve().parents[3] / "Issac-GR00T-N17"
        files = ("gr00t/eval/open_loop_eval.py", "gr00t/eval/run_gr00t_server.py",
                 "gr00t/policy/gr00t_policy.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in files:
                content = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=upstream)
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            ensure_adapter_inference(root)
            first = {name: (root / name).read_text() for name in files}
            ensure_adapter_inference(root)
            for name in files:
                self.assertEqual(first[name], (root / name).read_text())
                ast.parse(first[name])
            self.assertIn("load_model_for_inference", first[files[-1]])
            self.assertIn("base_model_path", first[files[0]])


if __name__ == "__main__":
    unittest.main()
