"""Batched differentiable FK for the exact RoboLab DROID gripper-base frame.

Joint frames are extracted from the pinned USD, including its Robotiq mount.
Runtime needs only torch and the small JSON asset, never Isaac Sim or USD.
"""

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

ASSET = Path(__file__).resolve().parents[1] / "assets/droid_kinematics.json"


def _frame(values):
    w, x, y, z = torch.tensor(values["quaternion_wxyz"], dtype=torch.float64)
    norm = torch.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = torch.stack((
        torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w))),
        torch.stack((2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w))),
        torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y))),
    ))
    result[:3, 3] = torch.tensor(values["position_m"], dtype=torch.float64)
    return result


class DroidForwardKinematics(nn.Module):
    def __init__(self):
        super().__init__()
        raw = ASSET.read_bytes()
        geometry = json.loads(raw)
        self.geometry_sha256 = hashlib.sha256(raw).hexdigest()
        self.asset_sha256 = geometry["asset_sha256"]
        if geometry["format"] != "robolab-droid-fk-v1" or geometry["units"] != "meters":
            raise ValueError("Unsupported DROID kinematics asset")
        self.axes = [joint["axis"] for joint in geometry["joints"]]
        if self.axes != ["X"] * 7 + [None, None]:
            raise ValueError("Expected the seven pinned DROID joint axes and two fixed joints")
        self.register_buffer("parent_frames", torch.stack([_frame(j["parent_frame"]) for j in geometry["joints"]]))
        self.register_buffer("child_inverses", torch.stack([torch.linalg.inv(_frame(j["child_frame"])) for j in geometry["joints"]]))

    def forward(self, joints):
        if joints.shape[-1] != 7:
            raise ValueError("DROID FK requires seven absolute joint angles")
        # Training uses bf16 autocast. Keep geometry in float32 (or caller's
        # float64) so a chain of nine transforms does not lose centimeter detail.
        with torch.autocast(device_type=joints.device.type, enabled=False):
            return self._positions(joints.float() if joints.dtype in (torch.float16, torch.bfloat16) else joints)

    def _positions(self, joints):
        parents, children = self.parent_frames.to(joints), self.child_inverses.to(joints)
        transform = torch.eye(4, dtype=joints.dtype, device=joints.device).expand(*joints.shape[:-1], 4, 4)
        for index, axis in enumerate(self.axes):
            transform = transform @ parents[index]
            if axis is not None:
                angle = joints[..., index]
                c, s = angle.cos(), angle.sin()
                one, zero = torch.ones_like(c), torch.zeros_like(c)
                rotation = torch.stack((
                    one, zero, zero, zero, zero, c, -s, zero,
                    zero, s, c, zero, zero, zero, zero, one,
                ), dim=-1).reshape(*angle.shape, 4, 4)
                transform = transform @ rotation
            transform = transform @ children[index]
        return transform[..., :3, 3]
