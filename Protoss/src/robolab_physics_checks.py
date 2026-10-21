"""Capture the first invalid physics state together with its preceding commands."""

from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _array(value):
    value = getattr(value, "torch", value)
    return value.detach().cpu().numpy().copy()


def _snapshot(robot, action):
    state = {"action": _array(action), "joint_position": _array(robot.data.joint_pos),
             "joint_velocity": _array(robot.data.joint_vel)}
    for name in ("root_pos_w", "root_quat_w", "applied_torque"):
        value = getattr(robot.data, name, None)
        if value is not None:
            state[name] = _array(value)
    return state


def _invalid_field(state):
    for name, value in state.items():
        if not np.isfinite(value).all():
            return name, tuple(int(i) for i in np.argwhere(~np.isfinite(value))[0])
    return None


def _raise_failure(env, robot, state, phase, step, history):
    from robolab.constants import get_output_dir

    name, index = _invalid_field(state)
    joint = (robot.data.joint_names[index[1]]
             if name in ("joint_position", "joint_velocity", "applied_torque") else None)
    values = {f"history_{key}": np.stack([sample[key] for sample in history]) for key in history[0]}
    values.update({f"fault_{key}": value for key, value in state.items()})
    values.update(joint_names=np.asarray(robot.data.joint_names), phase=np.asarray(phase), step=np.asarray(step),
                  physics_dt=np.asarray(env.cfg.sim.dt), seed=np.asarray(env.cfg.seed if env.cfg.seed is not None else -1))
    for key in ("soft_joint_pos_limits", "joint_vel_limits"):
        value = getattr(robot.data, key, None)
        if value is not None:
            values[key] = _array(value)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(get_output_dir()) / f"physics_failure-{stamp}-step{step}.npz"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **values)
        detail = f"Trace: {path}"
    except OSError as exc:
        detail = f"Trace save failed: {exc}"
    raise RuntimeError(f"Physics state became non-finite {phase}: step={step}, field={name}, "
                       f"index={index}, joint={joint}. {detail}")


def checked_step(env, action, advance):
    """Preserve actions; inspect state before/after physics and on step errors."""
    robot = env.scene["robot"]
    history = getattr(env, "_protoss_physics_history", None)
    if history is None:
        history = env._protoss_physics_history = deque(maxlen=1024)
    step = getattr(env, "_protoss_physics_step", 0) + 1
    env._protoss_physics_step = step
    before = _snapshot(robot, action)
    history.append(before)
    if _invalid_field(before):
        _raise_failure(env, robot, before, "before step", step, history)
    try:
        result = advance(action)
    except Exception:
        # A recorder/termination term may encounter the corrupt state first.
        failed = _snapshot(robot, action)
        if _invalid_field(failed):
            _raise_failure(env, robot, failed, "during step", step, history)
        raise
    after = _snapshot(robot, action)
    if _invalid_field(after):
        _raise_failure(env, robot, after, "after step", step, history)
    return result
