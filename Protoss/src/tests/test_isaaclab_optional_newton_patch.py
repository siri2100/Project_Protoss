"""Exercise the patched availability probe with real Python package discovery."""

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.isaaclab_optional_newton import SOURCE, ensure_optional_newton

ARTICULATION_HEADER = '''from isaaclab.utils.wrench_composer import WrenchComposer
_HAS_NEWTON_ACTUATORS = importlib.util.find_spec("isaaclab_newton.actuators") is not None


from isaaclab_physx.assets import kernels as shared_kernels
from isaaclab_physx.assets.articulation import kernels as articulation_kernels
from isaaclab_physx.physics import PhysxManager as SimulationManager
'''


class OptionalNewtonPatchTests(unittest.TestCase):
    def probe(self, package=None, broken=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_text(ARTICULATION_HEADER)
            self.assertTrue(ensure_optional_newton(root))
            self.assertFalse(ensure_optional_newton(root, check=True))
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

    def test_changes_only_assignment_when_surrounding_source_differs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            original = ("# Different surrounding source\r\nimport importlib.util\r\n\r\n"
                        "_HAS_NEWTON_ACTUATORS = importlib.util.find_spec('isaaclab_newton.actuators') is not None\r\n"
                        "\r\n# Keep custom edits\r\nCUSTOM = 42\r\n")
            source.write_bytes(original.encode())
            with self.assertRaisesRegex(ValueError, "Run setup"):
                ensure_optional_newton(root, check=True)
            self.assertEqual(source.read_bytes(), original.encode())
            self.assertTrue(ensure_optional_newton(root))
            changed = source.read_bytes()
            self.assertTrue(changed.startswith(b"# Different surrounding source\r\nimport importlib.util\r\n"))
            self.assertTrue(changed.endswith(b"# Keep custom edits\r\nCUSTOM = 42\r\n"))
            backup = source.with_suffix(".py.before_protoss_optional_newton")
            self.assertEqual(backup.read_bytes(), original.encode())
            self.assertFalse(ensure_optional_newton(root))
            self.assertEqual(source.read_bytes(), changed)
            self.assertEqual(backup.read_bytes(), original.encode())

    def test_unknown_probe_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / SOURCE
            source.parent.mkdir(parents=True)
            source.write_text('_HAS_NEWTON_ACTUATORS = custom_backend_check()\n')
            original = source.read_bytes()
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                ensure_optional_newton(root)
            self.assertEqual(source.read_bytes(), original)
            self.assertFalse(source.with_suffix(".py.before_protoss_optional_newton").exists())

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
