"""Keep the original smoke failure visible even if Kit exits during cleanup."""

import contextlib
import io
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src import robolab_595_smoke
from src import robolab_isaac60_compat


class SmokeFailureTests(unittest.TestCase):
    def run_failed_creation(self, exit_during_close):
        stderr, stdout = io.StringIO(), io.StringIO()

        def close():
            # The root cause must already be visible before cleanup begins.
            self.assertIn("RuntimeError: original scene creation error", stderr.getvalue())
            if exit_during_close:
                raise SystemExit(0)

        app = SimpleNamespace(close=Mock(side_effect=close))

        class Launcher:
            def __init__(self, args):
                self.app = app

            @staticmethod
            def add_app_launcher_args(parser):
                parser.add_argument("--device", default="cuda:0")

        modules = {}
        for name in (
            "cv2", "torch", "isaaclab", "isaaclab.app", "robolab", "robolab.constants",
            "robolab.registrations", "robolab.registrations.droid",
            "robolab.registrations.droid.auto_env_registrations_jointpos",
            "robolab.core", "robolab.core.environments",
            "robolab.core.environments.factory", "robolab.core.environments.runtime",
            "robolab.protoss_isaac60",
        ):
            modules[name] = ModuleType(name)
        modules["isaaclab.app"].AppLauncher = Launcher
        modules["robolab.protoss_isaac60"].CONTACT_FILTER_SETTING = "/physics/tensors/recursiveLeafPatternMatch"
        modules["robolab.protoss_isaac60"].as_torch = robolab_isaac60_compat.as_torch
        modules["robolab.protoss_isaac60"].get_backend_contact_sensors = robolab_isaac60_compat.get_backend_contact_sensors
        modules["robolab.constants"].set_output_dir = Mock()
        modules["robolab.registrations.droid.auto_env_registrations_jointpos"].auto_register_droid_envs = Mock()
        modules["robolab.core.environments.factory"].get_envs = Mock(return_value=["BananaOnPlateTask_0"])
        modules["robolab.core.environments.runtime"].create_env = Mock(
            side_effect=RuntimeError("original scene creation error"))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("sys.modules", modules), patch("sys.argv", ["smoke"]), \
                    patch.object(robolab_595_smoke.Path, "cwd", return_value=Path(directory)), \
                    contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                if exit_during_close:
                    with self.assertRaises(SystemExit) as caught:
                        robolab_595_smoke.main()
                    self.assertEqual(caught.exception.code, 1)
                else:
                    with self.assertRaisesRegex(RuntimeError, "original scene creation error"):
                        robolab_595_smoke.main()
        app.close.assert_called_once()
        self.assertIn("SMOKE FAILED at stage: environment_creation", stderr.getvalue())
        self.assertIn("Traceback", stderr.getvalue())
        self.assertNotIn("SMOKE PASS:", stdout.getvalue())

    def test_reports_original_exception_before_normal_shutdown(self):
        self.run_failed_creation(exit_during_close=False)

    def test_success_exit_from_shutdown_does_not_hide_scene_failure(self):
        self.run_failed_creation(exit_during_close=True)


if __name__ == "__main__":
    unittest.main()
