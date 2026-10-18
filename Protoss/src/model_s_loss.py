"""Model-S experiment: physical jerk of the one-step clean joint endpoint.

Edit ModelSConfig for defaults, joint_affine for representation/normalization,
and masked_jerk_loss for the regularizer. No GR00T source changes are required.
"""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ModelSConfig:
    jerk_weight: float = 1.0
    dt: float = 1 / 15  # DROID command interval, seconds; jerk is rad/s^3.
    gradient_diagnostics: bool = False  # Extra autograd passes on first microbatch.

    def __post_init__(self):
        if not math.isfinite(self.jerk_weight) or self.jerk_weight <= 0:
            raise ValueError("Model-S jerk weight must be finite and positive")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Model-S dt must be positive")


def config_from_args(args):
    return ModelSConfig(jerk_weight=args.s_jerk_weight,
                        dt=args.s_dt, gradient_diagnostics=args.s_gradient_diagnostics)


def joint_affine(processor, tag):
    """Read active pretrained normalization, never the fine-tuning subset stats.

    DROID joints are position offsets from ONE reference per chunk, not
    successive increments. Adding that constant reference does not change Δ³.
    Use affine inverse without clipping to preserve endpoint gradients outside
    [-1, 1]; deployed upstream min/max inverse clips, so this is a smooth surrogate.
    """
    sap = processor.state_action_processor
    modality = sap.modality_configs[tag]["action"]
    if modality.modality_keys != ["eef_9d", "gripper_position", "joint_position"]:
        raise ValueError("Model-S currently supports the DROID [eef_9d, gripper, joint] layout only")
    configs = modality.action_configs
    if not configs or configs[2].rep.value not in ("relative", "absolute"):
        raise ValueError("Model-S requires joint POSITION actions")
    if configs[2].type.value != "non_eef" or configs[2].format.value != "default":
        raise ValueError("Model-S requires plain joint position vectors")
    groups = sap.norm_params[tag]["action"]
    if int(groups["eef_9d"]["dim"]) != 9 or int(groups["gripper_position"]["dim"]) != 1:
        raise ValueError("Model-S requires EEF(9) and gripper(1) before the joints")
    params = groups["joint_position"]
    if int(params["dim"]) != 7:
        raise ValueError("Model-S expects seven DROID joints")
    if "joint_position" in (modality.mean_std_embedding_keys or []):
        std = torch.as_tensor(params["std"])
        mean = torch.as_tensor(params["mean"])
        # Match upstream's identity inverse for constant (std=0) dimensions.
        scale, offset = torch.where(std != 0, std, 1), torch.where(std != 0, mean, 0)
    else:
        # Active relative-action stats use min/max even when percentile mode is on.
        low, high = torch.as_tensor(params["min"]), torch.as_tensor(params["max"])
        scale, offset = (high - low) / 2, (high + low) / 2
    scale, offset = torch.as_tensor(scale).float(), torch.as_tensor(offset).float()
    # GR00T relative stats reduce over chunks, retaining the horizon axis.
    # Keep each timestep's own scale AND offset before taking differences.
    if (scale.shape != offset.shape or scale.ndim not in (1, 2)
            or scale.shape[-1] != 7 or scale.numel() == 0):
        raise ValueError(f"Model-S normalization must be (7,) or (H, 7); "
                         f"got scale={tuple(scale.shape)}, offset={tuple(offset.shape)}")
    if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or (scale < 0).any():
        raise ValueError("Invalid pretrained joint normalization")
    return scale, offset


def unnormalize_joints(normalized, mask, scale, offset):
    """Apply per-joint or per-horizon affine stats, respecting padded horizons."""
    scale, offset = scale.to(normalized), offset.to(normalized)
    if scale.ndim == 1 or scale.shape[0] == 1:
        return normalized * scale + offset
    horizon = normalized.shape[1]
    covered = min(horizon, scale.shape[0])
    if covered < horizon and mask[:, covered:].bool().any():
        raise ValueError(f"Model-S has {scale.shape[0]} normalization timesteps "
                         f"but valid joints extend into the {horizon}-step horizon")
    physical = normalized[:, :covered] * scale[:covered] + offset[:covered]
    if covered < horizon:
        # No valid actions here; don't invent normalization for padded timesteps.
        physical = torch.cat((physical, torch.zeros_like(normalized[:, covered:])), dim=1)
    return physical


def clean_action_endpoint(noisy_action, timestep, predicted_velocity):
    """GR00T uses x_t=(1-t)*noise+t*action and target velocity=action-noise."""
    t = timestep.float().reshape(-1, 1, 1)
    return noisy_action.float() + (1 - t) * predicted_velocity.float()


def masked_jerk_loss(joints, mask, dt):
    """Third finite difference; a window is valid only if all FOUR points are valid."""
    if joints.shape[1] < 4:
        raise ValueError("Model-S requires at least four action timesteps")
    valid = mask.bool()
    # Replace masked values before subtraction (0 * NaN is still NaN).
    values = torch.where(valid, joints.float(), 0.0)
    jerk = (values[:, 3:] - 3 * values[:, 2:-1]
            + 3 * values[:, 1:-2] - values[:, :-3]) / dt**3
    windows = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
    if not windows.any():
        raise ValueError("Model-S batch has no valid four-point joint windows")
    return torch.where(windows, jerk.square(), 0.0).sum() / windows.sum()


class ModelSLoss:
    """Jerk-only loss in physical units, without an RMS normalizer."""
    def __init__(self, config, scale, offset):
        self.config = config
        self.scale, self.offset = scale, offset

    def forward(self, model, inputs, diagnostics=False):
        from .model_s_forward import jerk_only_forward
        with jerk_only_forward(model.action_head):
            output = model(**inputs)
        clean = clean_action_endpoint(output["noisy"], output["t"], output["velocity"])
        mask = output["action_mask"][..., 10:17]
        joints = unnormalize_joints(clean[..., 10:17], mask, self.scale, self.offset)
        jerk = masked_jerk_loss(joints, mask, self.config.dt)
        total = self.config.jerk_weight * jerk
        if not torch.isfinite(total):
            raise ValueError("Non-finite Model-S jerk loss; stop before optimizer update")
        metrics = {"jerk_loss": jerk.detach().item(), "jerk_weight": self.config.jerk_weight,
                   "weighted_jerk_loss": total.detach().item()}
        if diagnostics and self.config.gradient_diagnostics:
            params = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(jerk, params, retain_graph=True, allow_unused=True)
            squared = torch.zeros((), device=jerk.device)
            for grad in grads:
                if grad is not None:
                    squared += grad.detach().float().square().sum()
            metrics["jerk_grad_norm"] = squared.sqrt().item()
        return total, metrics

    def state_dict(self):
        return {"format": "model-s-physical-jerk-only-v1", "config": asdict(self.config)}

    def load_state_dict(self, state):
        if state.get("format") != "model-s-physical-jerk-only-v1":
            raise ValueError("Previous Model-S loss requires a new physical jerk-only training run")
        if state["config"] != asdict(self.config):
            raise ValueError("Model-S resume loss configuration differs")
