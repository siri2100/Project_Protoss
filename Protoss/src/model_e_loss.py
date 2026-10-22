"""Model-E: preserve pretrained flow while shortening FK end-effector paths.

path_length_loss = batch_mean(sum_t ||FK(q[t+1]) - FK(q[t])||_2).
Joint targets are absolute, and EE path length is in meters. This loss has
no dt factor, squared distances, or target-based normalization.
"""

from dataclasses import asdict, dataclass
import math

import torch

from .model_s_forward import flow_preservation_forward
from .joint_actions import JointActionDecoder, clean_action_endpoint
from .droid_kinematics import DroidForwardKinematics


@dataclass(frozen=True)
class ModelEConfig:
    fm_weight: float = 1.0
    path_length_weight: float = 0.01

    def __post_init__(self):
        for name, value in (("FMD weight", self.fm_weight),
                            ("path length weight", self.path_length_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Model-E {name} must be finite and nonnegative")


def config_from_args(args):
    return ModelEConfig(fm_weight=args.e_fm_weight, path_length_weight=args.e_path_length_weight)


def masked_path_length_loss(positions, mask):
    """Mean EE chunk length; count only adjacent, fully observed XYZ positions.

    Padding and gaps never form a segment. Chunks with no valid segment are
    omitted from the batch mean. Invalid positions are masked BEFORE computing
    distances, so padded NaNs cannot contaminate values or gradients.
    """
    if positions.ndim != 3 or positions.shape != mask.shape or positions.shape[-1] != 3:
        raise ValueError("Model-E EE positions and mask must have matching (B, H, 3) shapes")
    if positions.shape[1] < 2:
        raise ValueError("Model-E requires at least two action timesteps")
    valid = mask.bool().all(dim=-1)
    segments = valid[:, 1:] & valid[:, :-1]
    chunks = segments.any(dim=1)
    if not chunks.any():
        raise ValueError("Model-E batch has no valid adjacent joint positions")
    values = torch.where(valid.unsqueeze(-1), positions, 0.0)
    delta = torch.where(segments.unsqueeze(-1), values[:, 1:] - values[:, :-1], 0.0)
    lengths = torch.linalg.vector_norm(delta, dim=-1).sum(dim=1)
    return lengths[chunks].mean()


class ModelELoss:
    def __init__(self, config, scale, offset, decoder=None):
        self.config = config
        self.scale, self.offset = scale, offset
        self.decoder = decoder if decoder is not None else JointActionDecoder(scale, offset)
        self.kinematics = DroidForwardKinematics()

    def forward(self, model, inputs, diagnostics=False):
        # Teacher/student share backbone features, noise and time, as in Model-S.
        model_inputs = inputs
        if "raw_joint_reference" in inputs.get("inputs", {}):
            # This reference belongs to the loss, not the pretrained model.
            model_inputs = {**inputs, "inputs": {key: value for key, value in inputs["inputs"].items()
                                                 if key != "raw_joint_reference"}}
        with flow_preservation_forward(model.action_head):
            output = model(**model_inputs)
        clean = clean_action_endpoint(output["noisy"], output["t"], output["velocity"])
        mask = output["action_mask"][..., 10:17]
        joints = self.decoder.absolute(clean[..., 10:17], mask, inputs)
        valid = mask.bool().all(dim=-1)
        # Sanitize padded joint vectors BEFORE nonlinear FK, not afterwards.
        joints = torch.where(valid[..., None], joints, 0.)
        if not torch.isfinite(joints).all():
            raise ValueError("Non-finite valid joint angles before FK")
        positions = self.kinematics(joints)
        ee_mask = valid[..., None].expand_as(positions)
        path_length = masked_path_length_loss(positions, ee_mask)
        fmd = output["loss"]
        weighted_fmd = self.config.fm_weight * fmd
        weighted_path = self.config.path_length_weight * path_length
        total = weighted_fmd + weighted_path
        if not torch.isfinite(total) or not torch.isfinite(path_length):
            raise ValueError("Non-finite Model-E FMD/path length loss")
        return total, {
            "fmd_loss": fmd.detach().item(),
            "weighted_fmd_loss": weighted_fmd.detach().item(),
            "fm_weight": self.config.fm_weight,
            "path_length_loss": path_length.detach().item(),
            "weighted_path_length_loss": weighted_path.detach().item(),
            "path_length_weight": self.config.path_length_weight,
            "ee_path_length": path_length.detach().item(),
        }

    def state_dict(self):
        return {"format": "model-e-flow-preservation-ee-path-length-v1", "config": asdict(self.config),
                "geometry_sha256": self.kinematics.geometry_sha256,
                "robot_asset_sha256": self.kinematics.asset_sha256}

    def load_state_dict(self, state):
        if state.get("format") != "model-e-flow-preservation-ee-path-length-v1":
            raise ValueError("Model-E EE path length requires a new training checkpoint; joint-path checkpoints cannot resume")
        if state.get("config") != asdict(self.config):
            raise ValueError("Model-E resume loss configuration differs")
        if state.get("geometry_sha256") != self.kinematics.geometry_sha256:
            raise ValueError("Model-E resume robot geometry differs")
