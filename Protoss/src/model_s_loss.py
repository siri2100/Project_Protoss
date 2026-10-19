"""Model-S: Pretrained flow preservation plus normalized joint velocity to encourage slower motion.

ModelSConfig: experimental settings. joint_affine: pretrained action inverse.
masked_velocity_loss: physical speed penalty. ModelSLoss: combined objective.
"""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ModelSConfig:
    velocity_lambda: float = 0.01
    velocity_reference: float | None = None  # Fixed RMS in rad/s; None estimates from targets.
    calibration_batches: int = 10
    velocity_reference_floor: float = 1e-3  # rad/s; guards stationary targets.
    dt: float = 1 / 15
    gradient_diagnostics: bool = False

    def __post_init__(self):
        if not math.isfinite(self.velocity_lambda) or self.velocity_lambda < 0:
            raise ValueError("Model-S velocity lambda must be finite and nonnegative")
        for name, value in (("dt", self.dt), ("velocity reference floor", self.velocity_reference_floor)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Model-S {name} must be finite and positive")
        if self.velocity_reference is not None and (not math.isfinite(self.velocity_reference) or self.velocity_reference <= 0):
            raise ValueError("Model-S velocity reference must be finite and positive")
        if self.calibration_batches < 1:
            raise ValueError("Model-S calibration batches must be positive")


def config_from_args(args):
    return ModelSConfig(velocity_lambda=args.s_velocity_lambda,
                        velocity_reference=args.s_velocity_reference,
                        calibration_batches=args.s_calibration_batches,
                        dt=args.s_dt, gradient_diagnostics=args.s_gradient_diagnostics)


def joint_affine(processor, tag):
    """Read active pretrained normalization, never the fine-tuning subset stats.

    DROID joints are position offsets from ONE reference per chunk, not
    successive increments. Adding that constant reference does not change Δq.
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


def velocity_terms(joints, mask, dt):
    """Sum of squared physical speeds and number of valid joint pairs."""
    if joints.shape[1] < 2:
        raise ValueError("Model-S requires at least two action timesteps")
    valid = mask.bool()
    values = torch.where(valid, joints.float(), 0.0)
    velocity = (values[:, 1:] - values[:, :-1]) / dt
    pairs = valid[:, 1:] & valid[:, :-1]
    count = pairs.sum()
    if not count:
        raise ValueError("Model-S batch has no valid joint pairs")
    return torch.where(pairs, velocity.square(), 0.0).sum(), count


def masked_velocity_loss(joints, mask, dt):
    total, count = velocity_terms(joints, mask, dt)
    return total / count


class ModelSLoss:
    """Pretrained-flow MSE + lambda * physical velocity MSE / fixed reference speed squared."""
    def __init__(self, config, scale, offset):
        self.config = config
        self.scale, self.offset = scale, offset
        self.reference = config.velocity_reference
        self.calibration_count = 0
        self.target_squared_sum = 0.0
        self.target_pair_count = 0

    @torch.no_grad()
    def calibrate(self, inputs):
        if self.reference is not None:
            return
        batch = inputs["inputs"]
        mask = batch["action_mask"][..., 10:17]
        joints = unnormalize_joints(batch["action"][..., 10:17].float(), mask, self.scale, self.offset)
        squared, count = velocity_terms(joints, mask, self.config.dt)
        if not torch.isfinite(squared):
            raise ValueError("Non-finite Model-S target velocity")
        self.target_squared_sum += squared.item()
        self.target_pair_count += count.item()
        self.calibration_count += 1
        if self.calibration_count >= self.config.calibration_batches:
            self.reference = max(math.sqrt(self.target_squared_sum / self.target_pair_count),
                                 self.config.velocity_reference_floor)
            print(f"Model-S velocity reference fixed: {self.reference:.8g} rad/s", flush=True)

    def forward(self, model, inputs, diagnostics=False):
        from .model_s_forward import model_s_forward
        if self.reference is None:
            raise ValueError("Calibrate Model-S velocity reference before training")
        # One shared backbone/noise/time supplies teacher and student flow predictions.
        with model_s_forward(model.action_head):
            output = model(**inputs)
        clean = clean_action_endpoint(output["noisy"], output["t"], output["velocity"])
        mask = output["action_mask"][..., 10:17]
        joints = unnormalize_joints(clean[..., 10:17], mask, self.scale, self.offset)
        raw_velocity = masked_velocity_loss(joints, mask, self.config.dt)
        normalized_velocity = raw_velocity / self.reference**2
        fm = output["loss"]
        weighted_velocity = self.config.velocity_lambda * normalized_velocity
        total = fm + weighted_velocity
        if not torch.isfinite(total) or not torch.isfinite(normalized_velocity):
            raise ValueError("Non-finite Model-S FM/velocity loss")
        metrics = {"fm_loss": fm.detach().item(), "raw_velocity_loss": raw_velocity.detach().item(),
                   "normalized_velocity_loss": normalized_velocity.detach().item(),
                   "weighted_velocity_loss": weighted_velocity.detach().item(),
                   "velocity_lambda": self.config.velocity_lambda,
                   "velocity_reference": self.reference}
        if diagnostics and self.config.gradient_diagnostics:
            params = [p for p in model.parameters() if p.requires_grad]
            for name, loss in (("fm", fm), ("velocity", normalized_velocity)):
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                squared = torch.zeros((), device=total.device)
                for grad in grads:
                    if grad is not None:
                        squared += grad.detach().float().square().sum()
                metrics[f"{name}_grad_norm"] = squared.sqrt().item()
        return total, metrics

    def state_dict(self):
        return {"format": "model-s-flow-preservation-velocity-v1", "config": asdict(self.config),
                "reference": self.reference, "calibration_count": self.calibration_count,
                "target_squared_sum": self.target_squared_sum, "target_pair_count": self.target_pair_count}

    def load_state_dict(self, state):
        if state.get("format") != "model-s-flow-preservation-velocity-v1":
            raise ValueError("Previous Model-S loss requires a new flow-preservation + velocity training run")
        if state["config"] != asdict(self.config):
            raise ValueError("Model-S resume loss configuration differs")
        for key in ("reference", "calibration_count", "target_squared_sum", "target_pair_count"):
            setattr(self, key, state[key])
