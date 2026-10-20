"""Load RoboLab's Python task configs without importing Isaac Lab's RL tasks.

Importing isaaclab_tasks registers its entire task suite, including Newton and
RL trainer dependencies. RoboLab only needs the Gym registry config factory.
"""

import importlib


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
