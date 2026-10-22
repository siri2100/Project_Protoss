"""FMD + joint SPARC numerical, gradient, normalization and PEFT integration checks."""
from contextlib import contextmanager
from enum import Enum
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.model_s_loss import (ModelSConfig, ModelSLoss, clean_action_endpoint,
                             joint_affine, unnormalize_joints)
from src.sparc_loss import masked_joint_sparc, spectral_arc_length
from src.model_s_forward import model_s_forward


class TwoArgLinear(nn.Linear):
    def forward(self, x, embodiment=None):
        return super().forward(x)


class TinyEncoder(nn.Module):
    def forward(self, noisy, t, embodiment):
        return noisy


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora = nn.Parameter(torch.tensor(.1))
        self.adapter_disabled = False
        self.calls = []

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False

    def forward(self, hidden_states, **kwargs):
        self.calls.append((hidden_states.detach().clone(), kwargs["timestep"].clone(),
                           self.adapter_disabled, torch.is_grad_enabled(), self.training))
        return hidden_states * (1 if self.adapter_disabled else 1 + self.lora), None


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = TinyEncoder()
        self.action_decoder = TwoArgLinear(20, 20, bias=False)
        self.state_encoder = TwoArgLinear(20, 20, bias=False)
        self.action_decoder.requires_grad_(False)
        self.state_encoder.requires_grad_(False)
        self.model = TinyDiT()
        self.config = SimpleNamespace(state_history_length=1, add_pos_embed=False,
                                      use_alternate_vl_dit=False)
        self.state_dropout_prob = 0
        self.num_timestep_buckets = 1000

    @property
    def lora(self):
        return self.model.lora

    def set_frozen_modules_to_eval_mode(self):
        pass

    def process_backbone_output(self, output):
        return output

    def sample_time(self, batch, **kwargs):
        return torch.full((batch,), .237, **kwargs)

    def forward(self, backbone, action_input):
        raise AssertionError("Model-S must use the shared flow forward")


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = TinyHead()

    def forward(self, inputs):
        action = inputs["action"]
        batch = action.shape[0]
        backbone = SimpleNamespace(backbone_features=torch.zeros(batch, 2, 20),
                                   backbone_attention_mask=torch.ones(batch, 2),
                                   image_mask=torch.ones(batch, 2))
        action_input = SimpleNamespace(action=action, action_mask=inputs["action_mask"],
                                       state=torch.zeros(batch, 1, 20),
                                       embodiment_id=torch.zeros(batch, dtype=torch.long))
        return self.action_head(backbone, action_input)


class ModelSLossTests(unittest.TestCase):
    def test_sparc_agrees_with_independent_numpy_reference_and_scale_invariance(self):
        import numpy as np
        t = np.arange(200) / 100.
        speed = np.exp(-5 * (t - 1)**2)
        nfft = 2 ** (int(np.ceil(np.log2(len(speed)))) + 4)
        frequency = np.fft.rfftfreq(nfft, d=.01)
        magnitude = np.abs(np.fft.rfft(speed, nfft))
        magnitude /= magnitude.max()
        indices = np.flatnonzero((frequency <= 10) & (magnitude >= .05))
        last = indices[-1]
        expected = -np.sqrt((np.diff(frequency[:last+1]) / frequency[last])**2
                            + np.diff(magnitude[:last+1])**2).sum()
        values = torch.tensor(speed, dtype=torch.float64, requires_grad=True)
        score = spectral_arc_length(values, .01)
        self.assertAlmostEqual(score.item(), expected, places=10)
        torch.testing.assert_close(spectral_arc_length(values * 7, .01), score)
        (-score).backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertTrue(torch.autograd.gradcheck(lambda x: spectral_arc_length(x, .01),
                                                (values.detach().requires_grad_(),), atol=1e-5))

    def test_sparc_prefers_smooth_profile_over_multiple_bursts(self):
        t = torch.linspace(0, 2, 201)
        smooth = torch.exp(-5 * (t-1).square())
        bursts = smooth * (1 + .8 * torch.cos(2 * torch.pi * 7 * t))
        self.assertGreater(spectral_arc_length(smooth, .01).item(),
                           spectral_arc_length(bursts, .01).item())

    def test_stationary_padding_and_gaps_have_finite_gradients(self):
        positions = torch.ones(1, 8, 7, requires_grad=True)
        score = masked_joint_sparc(positions, torch.ones_like(positions), 1/15)
        self.assertEqual(score.item(), 0.)
        score.backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        values = torch.zeros(1, 9, 7)
        values[:, :4] = torch.tensor([0., 1., 3., 4.])[None, :, None]
        values[:, 4] = float('nan')
        values[:, 5:] = torch.tensor([100., 101., 103., 104.])[None, :, None]
        mask = torch.ones_like(values)
        mask[:, 4] = 0
        values.requires_grad_()
        actual = masked_joint_sparc(values, mask, 1/15)
        expected = masked_joint_sparc(values[:, :4], mask[:, :4], 1/15)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual(values.grad[:, 4].abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, 'no valid'):
            masked_joint_sparc(values[:, :3], mask[:, :3], 1/15)

    def test_sparc_weight_scales_loss_and_lora_gradient_and_logs_components(self):
        torch.manual_seed(9)
        model = TinyPolicy()
        inputs = {'inputs': {'action': torch.randn(2, 8, 20), 'action_mask': torch.ones(2, 8, 20)}}
        rng = torch.get_rng_state()
        values, gradients = [], []
        for weight in (1., 10.):
            torch.set_rng_state(rng)
            objective = ModelSLoss(ModelSConfig(fm_weight=0, sparc_weight=weight, gradient_diagnostics=True),
                                  torch.ones(7), torch.zeros(7))
            total, metrics = objective.forward(model, inputs, diagnostics=True)
            values.append(total.detach())
            gradients.append(torch.autograd.grad(total, model.action_head.lora)[0])
            self.assertEqual(metrics['sparc_loss'], -metrics['sparc'])
            self.assertAlmostEqual(total.item(), metrics['weighted_fmd_loss'] + metrics['weighted_sparc_loss'])
            self.assertGreater(metrics['sparc_grad_norm'], 0)
        self.assertGreater(gradients[0].abs().item(), 0)
        torch.testing.assert_close(values[1], values[0] * 10)
        torch.testing.assert_close(gradients[1], gradients[0] * 10)

    def test_new_checkpoint_rejects_jerk_and_changed_settings(self):
        objective = ModelSLoss(ModelSConfig(), torch.ones(7), torch.zeros(7))
        state = objective.state_dict()
        objective.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'old jerk'):
            objective.load_state_dict({'format': 'model-s-flow-preservation-jerk-v1'})
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            ModelSLoss(ModelSConfig(sparc_weight=99), torch.ones(7), torch.zeros(7)).load_state_dict(state)
        for kwargs in ({'sparc_weight': -1}, {'sparc_weight': float('nan')}, {'dt': 0},
                       {'sparc_padlevel': -1}, {'sparc_cutoff_hz': 0}, {'sparc_amplitude_threshold': 1}):
            with self.assertRaises(ValueError):
                ModelSConfig(**kwargs)

    def test_fm_weight_scales_preservation_loss_and_gradient(self):
        model = TinyPolicy()
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        rng = torch.get_rng_state()
        values, grads, raw = [], [], []
        for weight in (1., 1000.):
            torch.set_rng_state(rng)
            objective = ModelSLoss(ModelSConfig(fm_weight=weight, sparc_weight=0), torch.ones(7), torch.zeros(7))
            total, metrics = objective.forward(model, inputs)
            values.append(total.detach())
            grads.append(torch.autograd.grad(total, model.action_head.lora)[0])
            raw.append(metrics['fmd_loss'])
            self.assertEqual(metrics['fm_weight'], weight)
            self.assertAlmostEqual(metrics['weighted_fmd_loss'], total.item())
        self.assertEqual(raw[0], raw[1])
        torch.testing.assert_close(values[1], values[0]*1000)
        torch.testing.assert_close(grads[1], grads[0]*1000)

    def test_teacher_failure_restores_adapter_training_flags_and_forward(self):
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(sparc_weight=.01), torch.ones(7), torch.zeros(7))
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        with patch.object(model.action_head.model, 'forward', side_effect=RuntimeError('teacher failed')):
            with self.assertRaisesRegex(RuntimeError, 'teacher failed'):
                objective.forward(model, inputs)
        self.assertFalse(model.action_head.model.adapter_disabled)
        self.assertTrue(model.action_head.model.training)
        self.assertNotIn('forward', model.action_head.__dict__)

    def test_real_peft_teacher_is_frozen_and_initial_distillation_is_zero(self):
        from peft import LoraConfig, get_peft_model

        class LinearDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(20, 20, bias=False)
                self.dropout = nn.Dropout(.7)
                with torch.no_grad():
                    self.proj.weight.copy_(torch.eye(20))

            def forward(self, hidden_states, **kwargs):
                return self.proj(self.dropout(hidden_states)), None

        model = TinyPolicy()
        model.action_head.model = get_peft_model(LinearDiT(), LoraConfig(
            r=2, lora_alpha=2, target_modules=['proj'], lora_dropout=0.))
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        objective = ModelSLoss(ModelSConfig(sparc_weight=.01), torch.ones(7), torch.zeros(7))
        captured = []
        handle = model.action_head.register_forward_hook(lambda module, args, output: captured.append(output))
        rng = torch.get_rng_state()
        try:
            total, first = objective.forward(model, inputs)
            self.assertEqual(first['fmd_loss'], 0.)
            self.assertFalse(captured[-1]['teacher_velocity'].requires_grad)
            total.backward()
            base = model.action_head.model.base_model.model.proj
            self.assertIsNone(base.base_layer.weight.grad)
            self.assertGreater(base.lora_B['default'].weight.grad.abs().sum().item(), 0)
            with torch.no_grad():
                base.lora_B['default'].weight.add_(.1)
            torch.set_rng_state(rng)
            _, second = objective.forward(model, inputs)
            torch.testing.assert_close(captured[0]['teacher_velocity'], captured[1]['teacher_velocity'])
            self.assertGreater(second['fmd_loss'], 0.)
            self.assertTrue(base.lora_B['default'].weight.requires_grad)
            self.assertTrue(model.action_head.model.base_model.model.dropout.training)
        finally:
            handle.remove()

    def test_true_velocity_recovers_clean_endpoint(self):
        torch.manual_seed(7)
        action, noise = torch.randn(3, 8, 17), torch.randn(3, 8, 17)
        t = torch.tensor([0., .237, 1.])
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action
        torch.testing.assert_close(clean_action_endpoint(noisy, t, action - noise), action)

    def test_excludes_gripper_eef_and_padded_dimensions(self):
        torch.manual_seed(8)
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(dt=1, sparc_weight=.01), torch.ones(7), torch.zeros(7))
        inputs = {"action": torch.randn(1, 8, 20), "action_mask": torch.ones(1, 8, 20)}
        # Diagonal decoder: excluded dimensions cannot affect joint predictions.
        with torch.no_grad():
            model.action_head.action_decoder.weight.copy_(torch.eye(20))
        rng = torch.get_rng_state()
        _, first = objective.forward(model, {"inputs": inputs})
        inputs["action"][..., :10] += 1000
        inputs["action"][..., 17:] += 1000
        torch.set_rng_state(rng)
        _, second = objective.forward(model, {"inputs": inputs})
        self.assertEqual(first["sparc_loss"], second["sparc_loss"])

    def test_joint_inverse_uses_active_stats_and_rejects_delta(self):
        class Rep(Enum):
            RELATIVE = "relative"
            DELTA = "delta"
        cfg = SimpleNamespace(rep=Rep.RELATIVE, type=SimpleNamespace(value="non_eef"), format=SimpleNamespace(value="default"))
        modality = SimpleNamespace(modality_keys=["eef_9d", "gripper_position", "joint_position"],
                                   action_configs=[None, None, cfg], mean_std_embedding_keys=None)
        params = {"dim": 7, "min": [-2.] * 7, "max": [4.] * 7}
        processor = SimpleNamespace(state_action_processor=SimpleNamespace(
            modality_configs={"droid": {"action": modality}}, norm_params={"droid": {"action": {
                "joint_position": params, "eef_9d": {"dim": 9}, "gripper_position": {"dim": 1}}}}))
        scale, offset = joint_affine(processor, "droid")
        torch.testing.assert_close(scale, torch.full((7,), 3.))
        torch.testing.assert_close(offset, torch.ones(7))
        # Relative stats preserve horizon, including timestep-dependent offsets.
        params["min"] = [[float(t)] * 7 for t in range(8)]
        params["max"] = [[float(t + 2)] * 7 for t in range(8)]
        scale, offset = joint_affine(processor, "droid")
        self.assertEqual(scale.shape, (8, 7))
        torch.testing.assert_close(offset[:, 0], torch.arange(1., 9.))
        cfg.rep = Rep.DELTA
        with self.assertRaisesRegex(ValueError, "POSITION"):
            joint_affine(processor, "droid")

    def test_horizon_normalization_recovers_physical_trajectory_and_gradient(self):
        t = torch.arange(8.).reshape(8, 1)
        scale = (1 + t).expand(8, 7)
        offset = (t**3).expand(8, 7)
        # Constant physical joint trajectory; normalized values are NOT smooth.
        expected = torch.full((2, 8, 7), 2.)
        normalized = ((expected - offset) / scale).requires_grad_()
        mask = torch.ones_like(normalized)
        physical = unnormalize_joints(normalized, mask, scale, offset)
        torch.testing.assert_close(physical, expected)
        torch.testing.assert_close(masked_joint_sparc(physical, mask, 1.), torch.tensor(0.), atol=1e-8, rtol=0)
        physical.sum().backward()
        torch.testing.assert_close(normalized.grad, scale.expand_as(normalized))

    def test_singleton_and_longer_stats_broadcast_and_slice(self):
        values = torch.randn(2, 8, 7)
        mask = torch.ones_like(values)
        for shape in ((7,), (1, 7), (8, 7), (10, 7)):
            physical = unnormalize_joints(values, mask, torch.full(shape, 2.), torch.ones(shape))
            torch.testing.assert_close(physical, 2 * values + 1)


if __name__ == "__main__":
    unittest.main()
