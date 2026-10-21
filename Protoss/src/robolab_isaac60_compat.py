"""Load RoboLab's Python task configs without importing Isaac Lab's RL tasks.

Importing isaaclab_tasks registers its entire task suite, including Newton and
RL trainer dependencies. RoboLab only needs the Gym registry config factory.
"""

import importlib


CONTACT_FILTER_SETTING = "/physics/tensors/recursiveLeafPatternMatch"


def wxyz_to_xyzw(quaternion):
    """Convert RoboLab-authored configuration values at the Lab 3 API boundary."""
    w, x, y, z = quaternion
    return (x, y, z, w)


def xyzw_to_wxyz(quaternion):
    x, y, z, w = quaternion
    return (w, x, y, z)


def get_backend_contact_sensors(scene):
    """Factories create backend subclasses of BaseContactSensor, not themselves."""
    from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor

    return {name: sensor for name, sensor in scene.sensors.items()
            if isinstance(sensor, BaseContactSensor)}


def as_torch(value):
    """Expose ProxyArray data to RoboLab before torch indexing or reductions."""
    return getattr(value, "torch", value)


def record_end_effector_state(robot, body_index):
    """Read Sim 6 data and preserve RoboLab's robot-root EE recording schema."""
    from isaaclab.utils.math import subtract_frame_transforms

    data = robot.data
    position, orientation = subtract_frame_transforms(
        as_torch(data.root_pos_w),
        as_torch(data.root_quat_w),
        as_torch(data.body_pos_w)[:, body_index, :],
        as_torch(data.body_quat_w)[:, body_index, :],
    )
    # Lab 3 math uses xyzw; RoboLab's recorded EE/root pose channels use wxyz.
    return {
        "position": position,
        "orientation": orientation.roll(1, dims=-1),
        "linear_velocity": as_torch(data.body_lin_vel_w)[:, body_index, :].clone(),
        "angular_velocity": as_torch(data.body_ang_vel_w)[:, body_index, :].clone(),
    }


def configure_contact_filters(sim_cfg):
    """Avoid the Sim 6.0.1 multi-collider contact-filter regression (#6424).

    Set through RenderCfg so it lands before PhysX creates its contact views.
    Keep other renderer settings and do not mutate a shared settings dictionary.
    """
    settings = dict(sim_cfg.render.carb_settings or {})
    settings[CONTACT_FILTER_SETTING] = False
    sim_cfg.render.carb_settings = settings


def load_cfg_from_registry(task_name, entry_point_key):
    import gymnasium as gym

    entry = gym.spec(task_name.split(":")[-1]).kwargs.get(entry_point_key)
    if entry is None:
        raise ValueError(f"Missing {entry_point_key} for {task_name}")
    if isinstance(entry, str):
        if entry.endswith((".yaml", ".yml")):
            raise ValueError("The Isaac Sim 6.0 port supports RoboLab Python task configs only")
        module, attribute = entry.split(":", 1)
        entry = getattr(importlib.import_module(module), attribute)
    return entry() if callable(entry) else entry
