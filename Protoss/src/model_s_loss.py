"""Model-S: weighted pretrained flow preservation plus normalized joint jerk.

The physical jerk MSE is divided by a fixed target RMS squared. Estimate that
reference before optimization, or supply it explicitly; never update it online.
"""
from dataclasses import asdict, dataclass
import math

import torch

from .joint_actions import clean_action_endpoint, joint_affine, unnormalize_actions as unnormalize_joints
from .model_s_forward import flow_preservation_forward


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


def jerk_terms(joints, mask, dt):
    """Sum squared physical jerk over windows with FOUR valid positions."""
    if joints.ndim != 3 or mask.shape != joints.shape:
        raise ValueError("Model-S jerk expects matching (B, H, D) positions/mask")
    if joints.shape[1] < 4:
        raise ValueError("Model-S requires at least four action timesteps")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Model-S jerk dt must be finite and positive")
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
        if self.reference is None:
            raise ValueError("Calibrate Model-S jerk reference before training")
        # One shared backbone/noise/time supplies teacher and student flow predictions.
        with flow_preservation_forward(model.action_head):
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
