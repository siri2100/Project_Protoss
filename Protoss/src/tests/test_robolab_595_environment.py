import tempfile
from pathlib import Path
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.robolab_595_environment import environment, write_project
from src.robolab_isaac60_compat import load_cfg_from_registry


class EnvironmentTests(unittest.TestCase):
    def test_loads_task_factory_once_without_importing_isaaclab_tasks(self):
        calls = []
        def factory():
            calls.append(True)
            return {"scene": "banana"}
        gym = SimpleNamespace(spec=lambda name: SimpleNamespace(kwargs={"env_cfg_entry_point": factory}))
        with patch.dict("sys.modules", {"gymnasium": gym}):
            self.assertEqual(load_cfg_from_registry("Banana", "env_cfg_entry_point"), {"scene": "banana"})
        self.assertEqual(len(calls), 1)

    def test_registry_errors_do_not_silently_use_another_task(self):
        gym = SimpleNamespace(spec=lambda name: SimpleNamespace(kwargs={}))
        with patch.dict("sys.modules", {"gymnasium": gym}):
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_cfg_from_registry("Banana", "env_cfg_entry_point")

    def test_preserves_workspace_sources_and_pins_sim_without_51_extra(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "IsaacLab").mkdir()
            (root / "IsaacLab" / "pyproject.toml").write_text('''
[project]
dependencies = ["isaaclab", "torch==2.10.0"]
[tool.uv]
package = false
pip = { index-strategy = "unsafe-best-match" }
[[tool.uv.index]]
url = "https://pypi.nvidia.com"
explicit = false
[tool.uv.sources]
isaaclab = { path = "source/isaaclab", editable = true }
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'linux'" }]
''')
            write_project(root)
            generated = tomllib.loads((root / "pyproject.toml").read_text())
            self.assertEqual(generated, environment(root))
            self.assertIn("isaacsim[all,extscache]==6.0.1.0", generated["project"]["dependencies"])
            self.assertIn("robolab", generated["project"]["dependencies"])
            self.assertIn("torch==2.11.0", generated["project"]["dependencies"])
            self.assertFalse(any("isaac51" in item for item in generated["project"]["dependencies"]))
            self.assertEqual(generated["tool"]["uv"]["sources"]["isaaclab"]["path"],
                             str((root / "IsaacLab/source/isaaclab").resolve()))
            self.assertTrue(generated["tool"]["uv"]["sources"]["robolab"]["editable"])
            self.assertEqual(generated["project"]["requires-python"], ">=3.12,<3.13")
            self.assertFalse(generated["tool"]["uv"]["package"])
            self.assertIn("x86_64", generated["tool"]["uv"]["environments"][0])


if __name__ == "__main__":
    unittest.main()
