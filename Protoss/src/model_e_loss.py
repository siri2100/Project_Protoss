"""Model-E: FMD + time-compressed teacher trajectory + joint motion limits."""

from dataclasses import asdict, dataclass
import math

import torch

from .joint_actions import JointActionDecoder, clean_action_endpoint, unnormalize_actions
from .model_s_forward import flow_preservation_forward


@dataclass(frozen=True)
class ModelEConfig:
    fm_weight: float = 1.0
    fast_weight: float = 1.0
    limit_weight: float = 0.01
    speedup: float = 1.2
    dt: float = 1 / 15
    velocity_limit: float = 2.0       # rad/s; experimental bound for each joint
    acceleration_limit: float = 10.0  # rad/s²; soft penalty, not a controller limit

    def __post_init__(self):
        for name in ("fm_weight", "fast_weight", "limit_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Model-E {name} must be finite and nonnegative")
        if not math.isfinite(self.speedup) or self.speedup < 1:
            raise ValueError("Model-E speedup must be finite and >= 1")
        for name in ("dt", "velocity_limit", "acceleration_limit"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Model-E {name} must be finite and positive")


def config_from_args(args):
    return ModelEConfig(fm_weight=args.e_fm_weight, fast_weight=args.e_fast_weight,
                        limit_weight=args.e_limit_weight, speedup=args.e_speedup,
                        dt=args.e_dt, velocity_limit=args.e_velocity_limit,
                        acceleration_limit=args.e_acceleration_limit)


def compressed_teacher_target(teacher, mask, speedup):
    """q_fast[k] = q_teacher[min(speedup*k, last)], with linear interpolation.

    Retiming happens in physical units, before fixed per-channel scaling. Each
    contiguous valid run has its own time origin; padding/gaps are never crossed.
    Runs need >=3 positions to offer a nontrivial speedup target. The teacher is
    always detached, and samples past the run's end hold its final position.
    """
    if teacher.ndim != 3 or mask.shape != teacher.shape:
        raise ValueError("Model-E trajectory/mask must have matching (B, H, D) shapes")
    if not math.isfinite(speedup) or speedup < 1:
        raise ValueError("Model-E speedup must be finite and >= 1")
    teacher = teacher.detach()
    if not torch.isfinite(torch.where(mask.bool(), teacher, 0.)).all():
        raise ValueError("Non-finite valid teacher trajectory")
    target, active = torch.zeros_like(teacher), torch.zeros_like(mask, dtype=torch.bool)
    flags = mask.bool().detach().cpu().tolist()
    for batch, frames in enumerate(flags):
        for channel in range(teacher.shape[-1]):
            start = None
            for index in range(teacher.shape[1] + 1):
                valid = index < len(frames) and frames[index][channel]
                if valid and start is None:
                    start = index
                elif not valid and start is not None:
                    length = index - start
                    if length >= 3:
                        times = (torch.arange(length, device=teacher.device, dtype=torch.float32)
                                 * speedup).clamp_max(length - 1)
                        left, right = times.long(), times.ceil().long()
                        fraction = times - left
                        run = teacher[batch, start:index, channel]
                        target[batch, start:index, channel] = torch.lerp(run[left], run[right], fraction)
                        active[batch, start:index, channel] = True
                    start = None
    return target, active


def masked_mean(values, mask):
    return torch.where(mask, values, 0.).sum() / mask.sum().clamp_min(1)


def joint_motion_limits(joints, mask, reference, config):
    """Dimensionless squared excess above joint velocity/acceleration bounds.

    Include current raw joint position -> first action in velocity. Acceleration
    compares adjacent predicted velocities, including that first interval; the
    previous measured velocity is not available here. Missing intervals are ignored.
    """
    if reference is None or reference.shape != (joints.shape[0], 7) or not torch.isfinite(reference).all():
        raise ValueError("Model-E limits require finite raw_joint_reference (B, 7)")
    valid = torch.cat((torch.ones_like(mask[:, :1], dtype=torch.bool), mask.bool()), dim=1)
    positions = torch.cat((reference.detach().to(joints)[:, None], joints), dim=1)
    segments = valid[:, 1:] & valid[:, :-1]
    velocity = torch.where(segments, (positions[:, 1:] - positions[:, :-1]) / config.dt, 0.)
    acceleration_mask = segments[:, 1:] & segments[:, :-1]
    acceleration = (velocity[:, 1:] - velocity[:, :-1]) / config.dt
    v_loss = masked_mean(torch.relu(velocity.abs() / config.velocity_limit - 1).square(), segments)
    a_loss = masked_mean(torch.relu(acceleration.abs() / config.acceleration_limit - 1).square(),
                         acceleration_mask)
    return v_loss, a_loss


def fixed_scale(scale):
    """Use one frozen scale per channel, even with horizon-dependent statistics."""
    scale = torch.as_tensor(scale).detach().float()
    scale = scale if scale.ndim == 1 else scale.amax(dim=0)
    return torch.where(scale > 1e-6, scale, 1.)


class ModelELoss:
    def __init__(self, config, scale, offset, decoder=None, gripper_scale=None, gripper_offset=None):
        self.config = config
        self.decoder = decoder if decoder is not None else JointActionDecoder(scale, offset)
        self.gripper_scale = torch.ones(1) if gripper_scale is None else gripper_scale
        self.gripper_offset = torch.zeros(1) if gripper_offset is None else gripper_offset
        self.normalizer = torch.cat((fixed_scale(scale), fixed_scale(self.gripper_scale)))

    def forward(self, model, inputs, diagnostics=False):
        features = inputs.get("inputs", {})
        # The raw reference belongs to the loss, not pretrained model features.
        model_inputs = {**inputs, "inputs": {k: v for k, v in features.items() if k != "raw_joint_reference"}}
        with flow_preservation_forward(model.action_head):
            output = model(**model_inputs)
        mask = output["action_mask"]
        motion_mask = torch.cat((mask[..., 10:17], mask[..., 9:10]), dim=-1).bool()

        def physical(velocity):
            clean = clean_action_endpoint(output["noisy"], output["t"], velocity)
            joints = self.decoder.absolute(clean[..., 10:17], mask[..., 10:17], inputs)
            gripper = unnormalize_actions(clean[..., 9:10], mask[..., 9:10],
                                        self.gripper_scale, self.gripper_offset)
            values = torch.where(motion_mask, torch.cat((joints, gripper), dim=-1), 0.)
            if not torch.isfinite(values).all():
                raise ValueError("Non-finite valid Model-E trajectory")
            return values

        student = physical(output["velocity"])
        with torch.no_grad():
            teacher = physical(output["teacher_velocity"])
            target, active = compressed_teacher_target(teacher, motion_mask, self.config.speedup)
        if not active[..., :7].any():
            raise ValueError("Model-E batch has no valid three-point joint runs for fast loss")
        error = ((student - target) / self.normalizer.to(student)).square()
        joint_fast = masked_mean(error[..., :7], active[..., :7])
        gripper_fast = masked_mean(error[..., 7:], active[..., 7:])
        fast = joint_fast + gripper_fast
        velocity, acceleration = joint_motion_limits(student[..., :7], motion_mask[..., :7],
                                                     features.get("raw_joint_reference"), self.config)
        limit = velocity + acceleration
        fmd = output["loss"]
        weighted_fmd, weighted_fast, weighted_limit = (self.config.fm_weight * fmd,
            self.config.fast_weight * fast, self.config.limit_weight * limit)
        total = weighted_fmd + weighted_fast + weighted_limit
        if not torch.isfinite(total):
            raise ValueError("Non-finite Model-E FMD/fast/limit loss")
        metrics = {"fmd_loss": fmd, "fast_loss": fast, "limit_loss": limit,
                   "joint_fast_loss": joint_fast, "gripper_fast_loss": gripper_fast,
                   "velocity_limit_loss": velocity, "acceleration_limit_loss": acceleration,
                   "weighted_fmd_loss": weighted_fmd, "weighted_fast_loss": weighted_fast,
                   "weighted_limit_loss": weighted_limit}
        return total, {**{k: v.detach().item() for k, v in metrics.items()},
                       "fm_weight": self.config.fm_weight, "fast_weight": self.config.fast_weight,
                       "limit_weight": self.config.limit_weight, "speedup": self.config.speedup}

    def state_dict(self):
        return {"format": "model-e-flow-preservation-fast-trajectory-v1", "config": asdict(self.config)}

    def load_state_dict(self, state):
        if state.get("format") != "model-e-flow-preservation-fast-trajectory-v1":
            raise ValueError("Model-E fast trajectory requires a new training run; old path-length checkpoints cannot resume")
        if state.get("config") != asdict(self.config):
            raise ValueError("Model-E resume loss configuration differs")
