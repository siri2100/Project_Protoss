"""Model-S: Pretrained flow preservation plus normalized joint jerk to encourage slower motion.

ModelSConfig: experimental settings. joint_affine: pretrained action inverse.
masked_jerk_loss: physical jerk penalty. ModelSLoss: combined objective.
"""
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ModelSConfig:
    jerk_lambda: float = 0.01
    jerk_reference: float | None = None  # Fixed RMS in rad/s^3; None estimates from targets.
    calibration_batches: int = 10
    jerk_reference_floor: float = 1.0  # rad/s^3; guards stationary targets.
    dt: float = 1 / 15
    gradient_diagnostics: bool = False
    fm_weight: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.fm_weight) or self.fm_weight < 0:
            raise ValueError("Model-S FM weight must be finite and nonnegative")
        if not math.isfinite(self.jerk_lambda) or self.jerk_lambda < 0:
            raise ValueError("Model-S jerk lambda must be finite and nonnegative")
        for name, value in (("dt", self.dt), ("jerk reference floor", self.jerk_reference_floor)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Model-S {name} must be finite and positive")
        if self.jerk_reference is not None and (not math.isfinite(self.jerk_reference) or self.jerk_reference <= 0):
            raise ValueError("Model-S jerk reference must be finite and positive")
        if self.calibration_batches < 1:
            raise ValueError("Model-S calibration batches must be positive")


def config_from_args(args):
    return ModelSConfig(fm_weight=args.s_fm_weight, jerk_lambda=args.s_jerk_lambda,
                        jerk_reference=args.s_jerk_reference,
                        calibration_batches=args.s_calibration_batches,
                        dt=args.s_dt, gradient_diagnostics=args.s_gradient_diagnostics)


def joint_affine(processor, tag):
    """Read active pretrained normalization, never the fine-tuning subset stats.

    DROID joints are position offsets from ONE reference per chunk, not
    successive increments. Adding that constant reference does not change Δ³q.
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
    """GR00T uses x_t=(1-t)*noise+t*action and target jerk=action-noise."""
    t = timestep.float().reshape(-1, 1, 1)
    return noisy_action.float() + (1 - t) * predicted_velocity.float()


def jerk_terms(joints, mask, dt):
    """Sum squared physical jerk over windows with FOUR valid positions."""
    if joints.shape[1] < 4:
        raise ValueError("Model-S requires at least four action timesteps")
    valid = mask.bool()
    values = torch.where(valid, joints.float(), 0.0)
    jerk = (values[:, 3:] - 3*values[:, 2:-1]
            + 3*values[:, 1:-2] - values[:, :-3]) / dt**3
    windows = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
    count = windows.sum()
    if not count:
        raise ValueError("Model-S batch has no valid four-point joint windows")
    return torch.where(windows, jerk.square(), 0.0).sum(), count


def masked_jerk_loss(joints, mask, dt):
    total, count = jerk_terms(joints, mask, dt)
    return total / count


class ModelSLoss:
    """Pretrained-flow MSE + lambda * physical jerk MSE / fixed reference jerk squared."""
    def __init__(self, config, scale, offset):
        self.config = config
        self.scale, self.offset = scale, offset
        self.reference = config.jerk_reference
        self.calibration_count = 0
        self.target_squared_sum = 0.0
        self.target_window_count = 0

    @torch.no_grad()
    def calibrate(self, inputs):
        if self.reference is not None:
            return
        batch = inputs["inputs"]
        mask = batch["action_mask"][..., 10:17]
        joints = unnormalize_joints(batch["action"][..., 10:17].float(), mask, self.scale, self.offset)
        squared, count = jerk_terms(joints, mask, self.config.dt)
        if not torch.isfinite(squared):
            raise ValueError("Non-finite Model-S target jerk")
        self.target_squared_sum += squared.item()
        self.target_window_count += count.item()
        self.calibration_count += 1
        if self.calibration_count >= self.config.calibration_batches:
            self.reference = max(math.sqrt(self.target_squared_sum / self.target_window_count),
                                 self.config.jerk_reference_floor)
            print(f"Model-S jerk reference fixed: {self.reference:.8g} rad/s^3", flush=True)

    def forward(self, model, inputs, diagnostics=False):
        from .model_s_forward import model_s_forward
        if self.reference is None:
            raise ValueError("Calibrate Model-S jerk reference before training")
        # One shared backbone/noise/time supplies teacher and student flow predictions.
        with model_s_forward(model.action_head):
            output = model(**inputs)
        clean = clean_action_endpoint(output["noisy"], output["t"], output["velocity"])
        mask = output["action_mask"][..., 10:17]
        joints = unnormalize_joints(clean[..., 10:17], mask, self.scale, self.offset)
        raw_jerk = masked_jerk_loss(joints, mask, self.config.dt)
        normalized_jerk = raw_jerk / self.reference**2
        fm = output["loss"]
        weighted_fm = self.config.fm_weight * fm
        weighted_jerk = self.config.jerk_lambda * normalized_jerk
        total = weighted_fm + weighted_jerk
        if not torch.isfinite(total) or not torch.isfinite(normalized_jerk):
            raise ValueError("Non-finite Model-S FMD/jerk loss")
        metrics = {"fmd_loss": fm.detach().item(), "raw_jerk_loss": raw_jerk.detach().item(),
                   "fm_weight": self.config.fm_weight, "weighted_fmd_loss": weighted_fm.detach().item(),
                   "normalized_jerk_loss": normalized_jerk.detach().item(),
                   "weighted_jerk_loss": weighted_jerk.detach().item(),
                   "jerk_lambda": self.config.jerk_lambda,
                   "jerk_reference": self.reference}
        if diagnostics and self.config.gradient_diagnostics:
            params = [p for p in model.parameters() if p.requires_grad]
            for name, loss in (("fmd", fm), ("jerk", normalized_jerk)):
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                squared = torch.zeros((), device=total.device)
                for grad in grads:
                    if grad is not None:
                        squared += grad.detach().float().square().sum()
                metrics[f"{name}_grad_norm"] = squared.sqrt().item()
        return total, metrics

    def state_dict(self):
        return {"format": "model-s-flow-preservation-jerk-v1", "config": asdict(self.config),
                "reference": self.reference, "calibration_count": self.calibration_count,
                "target_squared_sum": self.target_squared_sum, "target_window_count": self.target_window_count}

    def load_state_dict(self, state):
        if state.get("format") != "model-s-flow-preservation-jerk-v1":
            raise ValueError("Previous Model-S loss requires a new flow-preservation + jerk training run")
        saved_config = {"fm_weight": 1.0, **state["config"]}
        if saved_config != asdict(self.config):
            raise ValueError("Model-S resume loss configuration differs")
        for key in ("reference", "calibration_count", "target_squared_sum", "target_window_count"):
            setattr(self, key, state[key])
