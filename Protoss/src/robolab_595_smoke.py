"""Render and step one real RoboLab task without connecting to a policy server."""

import argparse
from pathlib import Path
import sys
import traceback


def report_failure(stage):
    # Kit shutdown may exit the process before Python prints an uncaught
    # exception. Emit it explicitly while the original exception is active.
    print(f"SMOKE FAILED at stage: {stage}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()


def main():
    import cv2  # noqa: F401 -- load before Isaac Lab, as in RoboLab's entrypoint
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="BananaOnPlateTask")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    app = AppLauncher(args).app
    env = None
    stage = "imports"
    failed = False
    try:
        import torch
        from robolab.constants import set_output_dir
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.core.environments.factory import get_envs
        from robolab.core.environments.runtime import create_env

        output = Path.cwd() / "output" / "protoss_isaac60_smoke"
        output.mkdir(parents=True, exist_ok=True)
        set_output_dir(str(output))
        stage = "task_registration"
        print(f"SMOKE stage: {stage}", flush=True)
        auto_register_droid_envs(task=[args.task])
        tasks = get_envs(task=[args.task])
        if not tasks:
            raise RuntimeError(f"Task not registered: {args.task}")
        stage = "environment_creation"
        print(f"SMOKE stage: {stage}", flush=True)
        env, cfg = create_env(tasks[0], device=args.device, seed=0, num_envs=1,
                              instruction_type=["default"], policy="gr00t")
        assert cfg.sim.physics.__class__.__name__ == "PhysxCfg"
        assert cfg.sim.physics.max_position_iteration_count == 32
        with torch.inference_mode():
            stage = "reset"
            print(f"SMOKE stage: {stage}", flush=True)
            env.reset()
            # Hold the initial pose; zero joint-position commands would move the arm.
            robot = env.scene["robot"]
            joints = [robot.data.joint_names.index(f"panda_joint{i}") for i in range(1, 8)]
            action = torch.cat((robot.data.joint_pos.torch[:, joints],
                                torch.zeros((1, 1), device=env.device)), dim=-1)
            if env.action_manager.total_action_dim != 8:
                raise RuntimeError("Smoke test expects DROID's seven arm joints + one gripper action")
            stage = "stepping"
            print(f"SMOKE stage: {stage}", flush=True)
            for _ in range(16):
                env.step(action)
            stage = "camera_validation"
            print(f"SMOKE stage: {stage}", flush=True)
            cameras = [sensor for sensor in env.scene.sensors.values()
                       if "rgb" in getattr(sensor.data, "output", {})]
            if not cameras:
                raise RuntimeError("No RGB camera rendered")
            for camera in cameras:
                rgb = camera.data.output["rgb"]
                rgb = getattr(rgb, "torch", rgb)
                if rgb.numel() == 0 or not torch.isfinite(rgb).all() or rgb.float().std() == 0:
                    raise RuntimeError("RGB camera produced an empty, invalid, or constant frame")
            print(f"SMOKE PASS: {args.task}, 16 steps, {len(cameras)} RGB cameras", flush=True)
    except BaseException:
        failed = True
        report_failure(stage)
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            try:
                app.close()
            except SystemExit:
                if failed:
                    raise SystemExit(1)
                raise


if __name__ == "__main__":
    main()
