"""Model-E: preserve pretrained flow while shortening joint-space paths.

path_length_loss = batch_mean(sum_t ||q[t+1] - q[t]||_2).
Joint positions use the pretrained affine inverse, in radians. This loss has
no dt factor, squared distances, or target-based normalization.
"""

from dataclasses import asdict, dataclass
import math

import torch

from .model_s_forward import flow_preservation_forward
from .model_s_loss import clean_action_endpoint, unnormalize_joints


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


def masked_path_length_loss(joints, mask):
    """Mean chunk length; count only adjacent, fully observed joint vectors.

    Padding and gaps never form a segment. Chunks with no valid segment are
    omitted from the batch mean. Invalid positions are masked BEFORE computing
    distances, so padded NaNs cannot contaminate values or gradients.
    """
    if joints.ndim != 3 or joints.shape != mask.shape:
        raise ValueError("Model-E joint positions and mask must have matching (B, H, D) shapes")
    if joints.shape[1] < 2:
        raise ValueError("Model-E requires at least two action timesteps")
    valid = mask.bool().all(dim=-1)
    segments = valid[:, 1:] & valid[:, :-1]
    chunks = segments.any(dim=1)
    if not chunks.any():
        raise ValueError("Model-E batch has no valid adjacent joint positions")
    values = torch.where(valid.unsqueeze(-1), joints.float(), 0.0)
    delta = torch.where(segments.unsqueeze(-1), values[:, 1:] - values[:, :-1], 0.0)
    lengths = torch.linalg.vector_norm(delta, dim=-1).sum(dim=1)
    return lengths[chunks].mean()


class ModelELoss:
    def __init__(self, config, scale, offset):
        self.config = config
        self.scale, self.offset = scale, offset

    def forward(self, model, inputs, diagnostics=False):
        # Teacher/student share backbone features, noise and time, as in Model-S.
        with flow_preservation_forward(model.action_head):
            output = model(**inputs)
        clean = clean_action_endpoint(output["noisy"], output["t"], output["velocity"])
        mask = output["action_mask"][..., 10:17]
        joints = unnormalize_joints(clean[..., 10:17], mask, self.scale, self.offset)
        path_length = masked_path_length_loss(joints, mask)
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
        }

    def state_dict(self):
        return {"format": "model-e-flow-preservation-path-length-v1", "config": asdict(self.config)}

    def load_state_dict(self, state):
        if state.get("format") != "model-e-flow-preservation-path-length-v1":
            raise ValueError("Model-E requires a flow-preservation + path-length training checkpoint")
        if state.get("config") != asdict(self.config):
            raise ValueError("Model-E resume loss configuration differs")
