"""Differentiable pretrained joint normalization and clean flow endpoint."""

import torch


def joint_affine(processor, tag):
    """Read active pretrained normalization, never the fine-tuning subset stats.

    DROID joints are position offsets from ONE reference per chunk, not
    successive increments. Adding that constant reference does not change joint velocities.
    Use affine inverse without clipping to preserve endpoint gradients outside
    [-1, 1]; deployed upstream min/max inverse clips, so this is a smooth surrogate.
    """
    sap = processor.state_action_processor
    modality = sap.modality_configs[tag]["action"]
    if modality.modality_keys != ["eef_9d", "gripper_position", "joint_position"]:
        raise ValueError("Model-S/E currently supports the DROID [eef_9d, gripper, joint] layout only")
    configs = modality.action_configs
    if not configs or configs[2].rep.value not in ("relative", "absolute"):
        raise ValueError("Model-S/E requires joint POSITION actions")
    if configs[2].type.value != "non_eef" or configs[2].format.value != "default":
        raise ValueError("Model-S/E requires plain joint position vectors")
    groups = sap.norm_params[tag]["action"]
    if int(groups["eef_9d"]["dim"]) != 9 or int(groups["gripper_position"]["dim"]) != 1:
        raise ValueError("Model-S/E requires EEF(9) and gripper(1) before the joints")
    params = groups["joint_position"]
    if int(params["dim"]) != 7:
        raise ValueError("Model-S/E expects seven DROID joints")
    return action_affine(processor, tag, "joint_position", 7)


def action_affine(processor, tag, key, dimension):
    """Read one action field's active inverse normalization, including horizon stats."""
    sap = processor.state_action_processor
    modality = sap.modality_configs[tag]["action"]
    params = sap.norm_params[tag]["action"][key]
    if int(params["dim"]) != dimension:
        raise ValueError(f"Expected {dimension} dimensions for {key}")
    if key in (modality.mean_std_embedding_keys or []):
        std, mean = torch.as_tensor(params["std"]), torch.as_tensor(params["mean"])
        scale, offset = torch.where(std != 0, std, 1), torch.where(std != 0, mean, 0)
    else:
        low, high = torch.as_tensor(params["min"]), torch.as_tensor(params["max"])
        scale, offset = (high - low) / 2, (high + low) / 2
    scale, offset = scale.float(), offset.float()
    if (scale.shape != offset.shape or scale.ndim not in (1, 2)
            or scale.shape[-1] != dimension or scale.numel() == 0):
        raise ValueError(f"Invalid {key} normalization shape: {tuple(scale.shape)}, {tuple(offset.shape)}")
    if not torch.isfinite(scale).all() or not torch.isfinite(offset).all() or (scale < 0).any():
        raise ValueError(f"Invalid pretrained {key} normalization")
    return scale, offset


def unnormalize_actions(normalized, mask, scale, offset):
    """Apply per-channel or per-horizon affine stats, respecting padded horizons."""
    scale, offset = scale.to(normalized), offset.to(normalized)
    if scale.ndim == 1 or scale.shape[0] == 1:
        return normalized * scale + offset
    horizon = normalized.shape[1]
    covered = min(horizon, scale.shape[0])
    if covered < horizon and mask[:, covered:].bool().any():
        raise ValueError(f"Model-S/E has {scale.shape[0]} normalization timesteps "
                         f"but valid actions extend into the {horizon}-step horizon")
    physical = normalized[:, :covered] * scale[:covered] + offset[:covered]
    if covered < horizon:
        # No valid actions here; don't invent normalization for padded timesteps.
        physical = torch.cat((physical, torch.zeros_like(normalized[:, covered:])), dim=1)
    return physical


def clean_action_endpoint(noisy_action, timestep, predicted_velocity):
    """GR00T uses x_t=(1-t)*noise+t*action and target flow=action-noise."""
    t = timestep.float().reshape(-1, 1, 1)
    return noisy_action.float() + (1 - t) * predicted_velocity.float()


class JointActionDecoder:
    """Restore absolute joint targets with the exact raw state used to encode them."""
    def __init__(self, scale, offset, relative=False):
        self.scale, self.offset = scale, offset
        self.relative = relative

    @classmethod
    def from_processor(cls, processor, tag):
        scale, offset = joint_affine(processor, tag)
        sap = processor.state_action_processor
        cfg = sap.modality_configs[tag]["action"].action_configs[2]
        relative = cfg.rep.value == "relative" and sap.use_relative_action
        return cls(scale, offset, relative)

    def absolute(self, normalized, mask, inputs):
        joints = unnormalize_actions(normalized, mask, self.scale, self.offset)
        if not self.relative:
            return joints
        reference = inputs["inputs"].get("raw_joint_reference")
        if reference is None or reference.shape != (joints.shape[0], 7):
            raise ValueError("Model-E requires raw_joint_reference (B, 7); normalized/clipped state is not an exact inverse")
        reference = reference.detach().to(joints)
        if not torch.isfinite(reference).all():
            raise ValueError("Model-E reference joints are non-finite")
        return joints + reference[:, None, :]
