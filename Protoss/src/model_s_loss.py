"""Model-S: weighted pretrained flow preservation plus joint speed SPARC loss."""

from dataclasses import asdict, dataclass
import math

import torch

from .joint_actions import clean_action_endpoint, joint_affine, unnormalize_actions as unnormalize_joints
from .model_s_forward import flow_preservation_forward
from .sparc_loss import masked_joint_sparc


@dataclass(frozen=True)
class ModelSConfig:
    fm_weight: float = 1.0
    sparc_weight: float = 0.01
    dt: float = 1 / 15
    sparc_padlevel: int = 4
    sparc_cutoff_hz: float = 10.0
    sparc_amplitude_threshold: float = 0.05
    gradient_diagnostics: bool = False

    def __post_init__(self):
        for name, value in (("FMD weight", self.fm_weight), ("SPARC weight", self.sparc_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Model-S {name} must be finite and nonnegative")
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Model-S dt must be positive and finite")
        if not isinstance(self.sparc_padlevel, int) or not 0 <= self.sparc_padlevel <= 10:
            raise ValueError("Model-S SPARC padlevel must be an integer in [0, 10]")
        if not math.isfinite(self.sparc_cutoff_hz) or self.sparc_cutoff_hz <= 0:
            raise ValueError("Model-S SPARC cutoff must be positive and finite")
        if not 0 < self.sparc_amplitude_threshold < 1:
            raise ValueError("Model-S SPARC amplitude threshold must be in (0, 1)")


def config_from_args(args):
    return ModelSConfig(fm_weight=args.s_fm_weight, sparc_weight=args.s_sparc_weight,
                        dt=args.s_dt, sparc_padlevel=args.s_sparc_padlevel,
                        sparc_cutoff_hz=args.s_sparc_cutoff_hz,
                        sparc_amplitude_threshold=args.s_sparc_amplitude_threshold,
                        gradient_diagnostics=args.s_gradient_diagnostics)


class ModelSLoss:
    """L = fm_weight * FMD + sparc_weight * (-mean joint SPARC)."""
    def __init__(self, config, scale, offset):
        self.config = config
        self.scale, self.offset = scale, offset

    def forward(self, model, inputs, diagnostics=False):
        with flow_preservation_forward(model.action_head):
            output = model(**inputs)
        clean = clean_action_endpoint(output["noisy"], output["t"], output["velocity"])
        mask = output["action_mask"][..., 10:17]
        # Relative joint offsets share a chunk reference; it cancels in velocity.
        joints = unnormalize_joints(clean[..., 10:17], mask, self.scale, self.offset)
        score = masked_joint_sparc(joints, mask, self.config.dt,
                                  self.config.sparc_padlevel, self.config.sparc_cutoff_hz,
                                  self.config.sparc_amplitude_threshold)
        sparc = -score
        fmd = output["loss"]
        weighted_fmd = self.config.fm_weight * fmd
        weighted_sparc = self.config.sparc_weight * sparc
        total = weighted_fmd + weighted_sparc
        if not torch.isfinite(total):
            raise ValueError("Non-finite Model-S FMD/SPARC loss")
        metrics = {"fmd_loss": fmd.detach().item(), "weighted_fmd_loss": weighted_fmd.detach().item(),
                   "fm_weight": self.config.fm_weight, "sparc": score.detach().item(),
                   "sparc_loss": sparc.detach().item(), "weighted_sparc_loss": weighted_sparc.detach().item(),
                   "sparc_weight": self.config.sparc_weight}
        if diagnostics and self.config.gradient_diagnostics:
            params = [p for p in model.parameters() if p.requires_grad]
            for name, loss in (("fmd", fmd), ("sparc", sparc)):
                grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                squared = torch.zeros((), device=total.device)
                for grad in grads:
                    if grad is not None:
                        squared += grad.detach().float().square().sum()
                metrics[f"{name}_grad_norm"] = squared.sqrt().item()
        return total, metrics

    def state_dict(self):
        return {"format": "model-s-flow-preservation-joint-sparc-v1", "config": asdict(self.config)}

    def load_state_dict(self, state):
        if state.get("format") != "model-s-flow-preservation-joint-sparc-v1":
            raise ValueError("Model-S SPARC requires a new training run; old jerk checkpoints cannot resume")
        if state.get("config") != asdict(self.config):
            raise ValueError("Model-S resume loss configuration differs")
