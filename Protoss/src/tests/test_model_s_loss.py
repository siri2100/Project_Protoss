"""Jerk-only Model-S: numerical loss, native-forward bypass and resume."""
from contextlib import contextmanager, nullcontext
from enum import Enum
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.model_s_loss import (ModelSConfig, ModelSLoss, clean_action_endpoint,
                             joint_affine, masked_jerk_loss, unnormalize_joints)
from src.model_s_forward import jerk_only_forward


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

    def forward(self, hidden_states, **kwargs):
        return hidden_states * self.lora, None


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
        raise AssertionError("Native FM forward must not run for Model-S")


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
    def test_cubic_endpoint_uses_physical_jerk_without_reference_division(self):
        model = TinyPolicy()
        clean = torch.zeros(1, 8, 20)
        clean[..., 10:17] = torch.arange(8.).reshape(1, 8, 1)**3
        output = dict(noisy=clean, velocity=torch.zeros_like(clean),
                      t=torch.zeros(1), action_mask=torch.ones_like(clean))
        objective = ModelSLoss(ModelSConfig(dt=.5), torch.ones(7), torch.zeros(7))
        with patch.object(model, 'forward', return_value=output):
            total, metrics = objective.forward(model, {'inputs': {}})
        self.assertEqual(metrics['jerk_loss'], 36 * 64)
        self.assertEqual(total.item(), 36 * 64)
        self.assertFalse(hasattr(objective, 'calibrate'))
        self.assertNotIn('jerk_reference', objective.state_dict())

    def test_true_velocity_recovers_clean_endpoint(self):
        torch.manual_seed(7)
        action, noise = torch.randn(3, 8, 17), torch.randn(3, 8, 17)
        t = torch.tensor([0., .237, 1.])
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action
        torch.testing.assert_close(clean_action_endpoint(noisy, t, action - noise), action)


    def test_jerk_cubic_and_dt_scaling(self):
        t = torch.arange(8.).reshape(1, -1, 1)
        for degree in range(3):
            self.assertEqual(masked_jerk_loss(t**degree, torch.ones_like(t), 1.).item(), 0.)
        self.assertEqual(masked_jerk_loss(t**3, torch.ones_like(t), .5).item(), 36 * 64)
        torch.testing.assert_close(masked_jerk_loss(t**3 + 100, torch.ones_like(t), 1.), torch.tensor(36.))


    def test_mask_requires_all_four_points_and_ignores_nan_padding(self):
        values = torch.arange(8.).reshape(1, -1, 1)**3
        mask = torch.ones_like(values)
        values[:, -1] = float("nan")
        mask[:, -1] = 0
        self.assertEqual(masked_jerk_loss(values, mask, 1.).item(), 36.)
        with self.assertRaisesRegex(ValueError, "no valid"):
            masked_jerk_loss(values, mask * 0, 1.)
        with self.assertRaisesRegex(ValueError, "four action"):
            masked_jerk_loss(values[:, :3], mask[:, :3], 1.)


    def test_excludes_gripper_eef_and_padded_dimensions(self):
        torch.manual_seed(8)
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(dt=1), torch.ones(7), torch.zeros(7))
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
        self.assertEqual(first["jerk_loss"], second["jerk_loss"])


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
        torch.testing.assert_close(masked_jerk_loss(physical, mask, 1.), torch.tensor(0.), atol=1e-8, rtol=0)
        physical.sum().backward()
        torch.testing.assert_close(normalized.grad, scale.expand_as(normalized))


    def test_short_stats_allow_only_masked_padding(self):
        values = torch.zeros(2, 10, 7)
        mask = torch.ones_like(values)
        mask[:, 8:] = 0
        scale, offset = torch.ones(8, 7), torch.arange(8.).reshape(8, 1).expand(8, 7)
        physical = unnormalize_joints(values, mask, scale, offset)
        self.assertEqual(physical.shape, values.shape)
        torch.testing.assert_close(masked_jerk_loss(physical, mask, 1.), torch.tensor(0.))
        mask[0, 8, 0] = 1
        with self.assertRaisesRegex(ValueError, "valid joints extend"):
            unnormalize_joints(values, mask, scale, offset)


    def test_singleton_and_longer_stats_broadcast_and_slice(self):
        values = torch.randn(2, 8, 7)
        mask = torch.ones_like(values)
        for shape in ((7,), (1, 7), (8, 7), (10, 7)):
            physical = unnormalize_joints(values, mask, torch.full(shape, 2.), torch.ones(shape))
            torch.testing.assert_close(physical, 2 * values + 1)


    def test_no_fm_forward_loss_or_gradient_and_both_dit_branches(self):
        model = TinyPolicy()
        inputs = {"inputs": {"action": torch.randn(2, 8, 20), "action_mask": torch.ones(2, 8, 20)}}
        for alternate in (False, True):
            model.action_head.config.use_alternate_vl_dit = alternate
            objective = ModelSLoss(ModelSConfig(gradient_diagnostics=True), torch.ones(7), torch.zeros(7))
            with patch('torch.nn.functional.mse_loss', side_effect=AssertionError('FM MSE called')):
                total, metrics = objective.forward(model, inputs, diagnostics=True)
            self.assertNotIn('fm_loss', metrics)
            self.assertNotIn('fm_weight', metrics)
            self.assertGreater(metrics['jerk_grad_norm'], 0)
            self.assertEqual(total.item(), metrics['raw_jerk_loss'])
            total.backward()
            self.assertTrue(torch.isfinite(model.action_head.lora.grad))
            self.assertIsNone(model.action_head.action_decoder.weight.grad)
            self.assertNotIn('forward', model.action_head.__dict__)

    def test_raw_jerk_is_exact_training_tensor_and_gradient(self):
        model = TinyPolicy()
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        objective = ModelSLoss(ModelSConfig(dt=1), torch.ones(7), torch.zeros(7))
        from src.model_s_loss import masked_jerk_loss as calculate
        captured = []
        def capture(*args):
            result = calculate(*args)
            captured.append(result)
            return result
        with patch('src.model_s_loss.masked_jerk_loss', side_effect=capture):
            loss, metrics = objective.forward(model, inputs)
        self.assertIs(loss, captured[0])
        self.assertEqual(loss.item(), metrics['raw_jerk_loss'])
        self.assertEqual(metrics['jerk_loss'], metrics['raw_jerk_loss'])
        expected_grad = torch.autograd.grad(captured[0], model.action_head.lora, retain_graph=True)[0]
        actual_grad = torch.autograd.grad(loss, model.action_head.lora)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        state = objective.state_dict()
        state['config']['jerk_weight'] = 1.0
        objective.load_state_dict(state)
        state['config']['jerk_weight'] = .25
        with self.assertRaisesRegex(ValueError, 'new raw jerk'):
            objective.load_state_dict(state)

    def test_forward_restored_on_failure(self):
        head = TinyHead()
        with self.assertRaises(RuntimeError):
            with jerk_only_forward(head):
                raise RuntimeError('failed')
        self.assertNotIn('forward', head.__dict__)

    def test_training_loop_uses_physical_jerk_from_first_update(self):
        import train as entry
        from src import training_runtime as runtime
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(dt=1), torch.ones(7), torch.zeros(7))
        args = SimpleNamespace(max_steps=2, gradient_accumulation=2, save_steps=2, no_hf_upload=True)
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        session = SimpleNamespace(model=model, step=0, smoothness=objective,
            batches=lambda count: [inputs] * count, autocast=nullcontext,
            trainer=SimpleNamespace(state=SimpleNamespace(log_history=[])))
        @contextmanager
        def setup(*args):
            yield session
        with patch.object(entry, 'prepare_training', return_value=(None, {})), \
             patch.object(runtime, 'setup_training', setup), \
             patch.object(runtime, 'preprocess', side_effect=lambda batch, _: batch), \
             patch.object(runtime, 'update_weights') as update, \
             patch.object(runtime, 'save_checkpoint'), patch.object(runtime, 'export_model'), \
             patch.object(runtime, 'save_training_loss_plot'), \
             patch.object(runtime, 'save_training_distribution_plot'):
            entry.main(args)
        self.assertEqual(update.call_count, 2)
        for row in session.trainer.state.log_history:
            self.assertNotIn('fm_loss', row)
            self.assertGreater(row['jerk_loss'], 0)
            self.assertAlmostEqual(row['loss'], row['raw_jerk_loss'], places=5)
