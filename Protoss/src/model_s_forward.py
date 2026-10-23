"""Model-FMD/S/E forward: pretrained flow distillation and clean endpoint inputs.

The noise/time/DiT prediction is shared by FMD and the motion objectives.
"""
from contextlib import contextmanager
from types import MethodType

import torch


@contextmanager
def deterministic_dit(dit):
    """Disable dropout for both passes and restore every module's training flag."""
    states = [(module, module.training) for module in dit.modules()]
    dit.eval()
    try:
        yield
    finally:
        for module, training in states:
            module.training = training


def flow_action_head_forward(head, backbone_output, action_input):
    head.set_frozen_modules_to_eval_mode()
    backbone_output = head.process_backbone_output(backbone_output)
    vl_embeds = backbone_output.backbone_features
    embodiment_id = action_input.embodiment_id

    assert action_input.state.shape[1] == head.config.state_history_length
    state = action_input.state.reshape(action_input.state.shape[0], 1, -1)
    state_features = head.state_encoder(state, embodiment_id)
    if head.training and head.state_dropout_prob > 0:
        dropped = torch.rand(state_features.shape[0], device=state_features.device) < head.state_dropout_prob
        state_features = state_features * (~dropped[:, None, None]).to(state_features.dtype)

    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = head.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]
    noisy = (1 - t) * noise + t * actions
    discrete_t = (t[:, 0, 0] * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy, discrete_t, embodiment_id)
    if head.config.add_pos_embed:
        positions = torch.arange(action_features.shape[1], device=action_features.device)
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)

    kwargs = dict(hidden_states=torch.cat((state_features, action_features), dim=1),
                  encoder_hidden_states=vl_embeds,
                  encoder_attention_mask=backbone_output.backbone_attention_mask,
                  timestep=discrete_t, return_all_hidden_states=True)
    if head.config.use_alternate_vl_dit:
        kwargs.update(image_mask=backbone_output.image_mask,
                      backbone_attention_mask=backbone_output.backbone_attention_mask)
    if not callable(getattr(head.model, "disable_adapter", None)):
        raise ValueError("Flow preservation teacher requires a PEFT DiT with disable_adapter()")
    # Frozen pretrained weights are the teacher: no second backbone/model copy.
    # Both passes receive the SAME hidden states, conditioning and discrete time.
    with deterministic_dit(head.model):
        with torch.no_grad(), head.model.disable_adapter():
            teacher_hidden, _ = head.model(**kwargs)
            teacher_velocity = head.action_decoder(teacher_hidden, embodiment_id)[:, -actions.shape[1]:]
        hidden, _ = head.model(**kwargs)
        velocity = head.action_decoder(hidden, embodiment_id)[:, -actions.shape[1]:]
    mask = action_input.action_mask
    action_loss = (velocity.float() - teacher_velocity.float().detach()).square() * mask
    fmd_loss = action_loss.sum() / (mask.sum() + 1e-6)
    return {"noisy": noisy, "t": t, "velocity": velocity,
            "teacher_velocity": teacher_velocity.detach(),
            "action_mask": mask, "loss": fmd_loss}


@contextmanager
def flow_preservation_forward(head):
    """Use teacher/student flow during training; restore native inference forward."""
    had_override = "forward" in head.__dict__
    original = head.forward
    head.forward = MethodType(flow_action_head_forward, head)
    try:
        yield
    finally:
        if had_override:
            head.forward = original
        else:
            del head.forward


# Preserve the existing Model-S import while sharing the same implementation.
model_s_forward = flow_preservation_forward
