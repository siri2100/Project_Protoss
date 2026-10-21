"""Exercise the patched availability probe with real Python package discovery."""

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ARTICULATION_HEADER = '''from isaaclab.utils.wrench_composer import WrenchComposer
_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None


from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.assets.articulation import kernels as articulation_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager
'''


class OptionalNewtonPatchTests(unittest.TestCase):
    def probe(self, package=None, broken=False):
        patch = Path(__file__).resolve().parents[2] / "patches/isaaclab_60_optional_newton.patch"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py"
            source.parent.mkdir(parents=True)
            source.write_text(ARTICULATION_HEADER)
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "apply", str(patch)], cwd=root, check=True, capture_output=True)
            subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                           cwd=root, check=True, capture_output=True)
            tree = ast.parse(source.read_text())
            probe = next(node for node in tree.body if isinstance(node, ast.Assign))
            if package:
                backend = root / "isaaclab_newton"
                backend.mkdir()
                (backend / "__init__.py").write_text('raise RuntimeError("broken backend")' if broken else "")
                if package == "actuators":
                    (backend / "actuators.py").write_text("")
            code = "import importlib.util\n" + ast.unparse(probe) + "\nprint(_HAS_NEWTON_ACTUATORS)\n"
            return subprocess.run([sys.executable, "-S", "-c", code], cwd=root,
                                  text=True, capture_output=True)

    def test_no_newton_package_disables_optional_actuators(self):
        result = self.probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_parent_without_actuators_disables_optional_actuators(self):
        result = self.probe(package="parent")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_available_actuators_remain_enabled(self):
        result = self.probe(package="actuators")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "True")

    def test_broken_installed_package_is_not_silently_hidden(self):
        result = self.probe(package="actuators", broken=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RuntimeError: broken backend", result.stderr)


if __name__ == "__main__":
    unittest.main()
