"""GR00T N1.7 training forward for Model-S, without FM targets or FM MSE.

Keep the pretrained feature path, noise interpolation and decoder unchanged.
Return the clean endpoint inputs for the jerk objective instead of an FM loss.
"""
from contextlib import contextmanager
from types import MethodType

import torch


def jerk_action_head_forward(head, backbone_output, action_input):
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
    hidden, _ = head.model(**kwargs)
    velocity = head.action_decoder(hidden, embodiment_id)[:, -actions.shape[1]:]
    return {"noisy": noisy, "t": t, "velocity": velocity,
            "action_mask": action_input.action_mask}


@contextmanager
def jerk_only_forward(head):
    """Restore native forward even on failure; inference and Model-FM are unaffected."""
    had_override = "forward" in head.__dict__
    original = head.forward
    head.forward = MethodType(jerk_action_head_forward, head)
    try:
        yield
    finally:
        if had_override:
            head.forward = original
        else:
            del head.forward
