"""Model-S experiment: native FM + jerk of the one-step clean joint endpoint.

Edit ModelSConfig for defaults, joint_affine for representation/normalization,
and masked_jerk_loss for the regularizer. No GR00T source changes are required.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math

import torch


@dataclass(frozen=True)
class ModelSConfig:
    # None calibrates lambda = ratio * fm_weight * mean(FM) / mean(jerk).
    jerk_weight: float | None = None
    jerk_ratio: float = 0.01
    calibration_steps: int = 10
    dt: float = 1 / 15  # DROID command interval, seconds; jerk is rad/s^3.
    gradient_diagnostics: bool = False  # Extra autograd passes on first microbatch.
    fm_weight: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.fm_weight) or self.fm_weight < 0:
            raise ValueError("Model-S FM weight must be finite and nonnegative")
        if self.fm_weight == 0 and self.jerk_weight is None:
            raise ValueError("Zero FM weight requires an explicit --s-jerk-weight")
        if self.jerk_weight is not None and (not math.isfinite(self.jerk_weight) or self.jerk_weight < 0):
            raise ValueError("Model-S jerk weight must be finite and nonnegative")
        if not math.isfinite(self.jerk_ratio) or self.jerk_ratio <= 0:
            raise ValueError("Model-S jerk ratio must be finite and positive")
        if self.calibration_steps < 1 or not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Model-S calibration steps and dt must be positive")


def config_from_args(args):
    return ModelSConfig(jerk_weight=args.s_jerk_weight, jerk_ratio=args.s_jerk_ratio,
                        calibration_steps=args.s_calibration_steps, dt=args.s_dt,
                        gradient_diagnostics=args.s_gradient_diagnostics,
                        fm_weight=args.s_fm_weight)


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


@contextmanager
def capture_flow_forward(head):
    """Capture exact continuous t, x_t and decoded velocity from the SAME forward.

    Hooks are removed before backward, including gradient checkpoint recomputation.
    Nothing is detached: jerk gradients flow through the frozen decoder to LoRA.
    """
    captured = {}
    original = head.sample_time
    had_override = "sample_time" in head.__dict__

    def sample_time(*args, **kwargs):
        t = original(*args, **kwargs)
        captured["t"] = t
        return t

    def capture_noisy(module, args):
        captured["noisy"] = args[0]

    def capture_velocity(module, args, output):
        captured["velocity"] = output

    head.sample_time = sample_time
    handles = []
    try:
        handles.append(head.action_encoder.register_forward_pre_hook(capture_noisy))
        handles.append(head.action_decoder.register_forward_hook(capture_velocity))
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        if had_override:
            head.sample_time = original
        else:
            del head.sample_time


class ModelSLoss:
    """Own calibration state explicitly so resume keeps exactly the same lambda."""
    def __init__(self, config, scale, offset):
        self.config = config
        self.scale, self.offset = scale, offset
        self.weight = config.jerk_weight
        self.calibration_count = 0
        self.fm_sum = self.jerk_sum = 0.0

    def forward(self, model, inputs, diagnostics=False):
        with capture_flow_forward(model.action_head) as flow:
            output = model(**inputs)
        if set(flow) != {"t", "noisy", "velocity"}:
            raise RuntimeError("GR00T flow forward changed: missing endpoint inputs")
        horizon = flow["noisy"].shape[1]
        velocity = flow["velocity"][:, -horizon:]
        clean = clean_action_endpoint(flow["noisy"], flow["t"], velocity)
        # DROID: EEF 0:9, gripper 9, joint positions 10:17, then padding.
        mask = output["action_mask"][..., 10:17]
        joints = unnormalize_joints(clean[..., 10:17], mask, self.scale, self.offset)
        jerk = masked_jerk_loss(joints, mask, self.config.dt)
        fm = output["loss"]  # Keep upstream masking and reduction unchanged.
        weight = self.weight if self.weight is not None else 0.0
        total = self.config.fm_weight * fm + weight * jerk
        if not torch.isfinite(total) or not torch.isfinite(jerk):
            raise ValueError("Non-finite Model-S loss; stop before optimizer update")
        metrics = {"fm_loss": fm.detach().item(), "jerk_loss": jerk.detach().item(),
                   "fm_weight": self.config.fm_weight,
                   "weighted_fm_loss": self.config.fm_weight * fm.detach().item(),
                   "jerk_weight": weight, "weighted_jerk_loss": weight * jerk.detach().item()}
        if diagnostics and self.config.gradient_diagnostics:
            params = [p for p in model.parameters() if p.requires_grad]
            for name, loss in (("fm", fm), ("jerk", jerk)):
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                squared_norm = torch.zeros((), device=fm.device)
                for grad in grads:
                    if grad is not None:
                        squared_norm += grad.detach().float().square().sum()
                metrics[f"{name}_grad_norm"] = squared_norm.sqrt().item()
        return total, metrics

    def finish_step(self, metrics):
        """Collect optimizer-step means; regularization starts AFTER calibration."""
        if self.weight is not None:
            return
        self.fm_sum += metrics["fm_loss"]
        self.jerk_sum += metrics["jerk_loss"]
        self.calibration_count += 1
        if self.calibration_count >= self.config.calibration_steps:
            if self.jerk_sum <= 0:
                raise ValueError("Cannot calibrate Model-S from zero jerk; set --s-jerk-weight explicitly")
            self.weight = self.config.jerk_ratio * self.config.fm_weight * self.fm_sum / self.jerk_sum
            print(f"Model-S calibration complete: fixed jerk_weight={self.weight:.8g}", flush=True)

    def state_dict(self):
        return {"config": asdict(self.config), "weight": self.weight,
                "calibration_count": self.calibration_count,
                "fm_sum": self.fm_sum, "jerk_sum": self.jerk_sum}

    def load_state_dict(self, state):
        # Older Model-S runs used an implicit FM weight of one.
        saved_config = {"fm_weight": 1.0, **state["config"]}
        if saved_config != asdict(self.config):
            raise ValueError("Model-S resume loss configuration differs")
        for key in ("weight", "calibration_count", "fm_sum", "jerk_sum"):
            setattr(self, key, state[key])
