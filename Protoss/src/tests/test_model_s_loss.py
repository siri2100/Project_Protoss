"""Model-S FM plus normalized velocity: CPU numerical and integration tests."""
from contextlib import contextmanager, nullcontext
from enum import Enum
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.model_s_loss import (ModelSConfig, ModelSLoss, clean_action_endpoint,
                             joint_affine, masked_velocity_loss, unnormalize_joints)
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
    def test_teacher_failure_restores_adapter_training_flags_and_forward(self):
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(velocity_reference=1), torch.ones(7), torch.zeros(7))
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
        objective = ModelSLoss(ModelSConfig(velocity_reference=1), torch.ones(7), torch.zeros(7))
        captured = []
        handle = model.action_head.register_forward_hook(lambda module, args, output: captured.append(output))
        rng = torch.get_rng_state()
        try:
            total, first = objective.forward(model, inputs)
            self.assertEqual(first['fm_loss'], 0.)
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
            self.assertGreater(second['fm_loss'], 0.)
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
        objective = ModelSLoss(ModelSConfig(dt=1, velocity_reference=1), torch.ones(7), torch.zeros(7))
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
        self.assertEqual(first["normalized_velocity_loss"], second["normalized_velocity_loss"])


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
        torch.testing.assert_close(masked_velocity_loss(physical, mask, 1.), torch.tensor(0.), atol=1e-8, rtol=0)
        physical.sum().backward()
        torch.testing.assert_close(normalized.grad, scale.expand_as(normalized))


    def test_singleton_and_longer_stats_broadcast_and_slice(self):
        values = torch.randn(2, 8, 7)
        mask = torch.ones_like(values)
        for shape in ((7,), (1, 7), (8, 7), (10, 7)):
            physical = unnormalize_joints(values, mask, torch.full(shape, 2.), torch.ones(shape))
            torch.testing.assert_close(physical, 2 * values + 1)


    def test_velocity_linear_trajectory_dt_mask_and_padding(self):
        t = torch.arange(5.).reshape(1, 5, 1)
        self.assertEqual(masked_velocity_loss(2*t, torch.ones_like(t), .5).item(), 16.)
        self.assertEqual(masked_velocity_loss(t*0+5, torch.ones_like(t), .5).item(), 0.)
        t[:, -1] = float('nan')
        mask = torch.ones_like(t); mask[:, -1] = 0
        self.assertEqual(masked_velocity_loss(t, mask, 1).item(), 1.)
        with self.assertRaisesRegex(ValueError, 'no valid joint pairs'):
            masked_velocity_loss(t, mask*0, 1)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            masked_velocity_loss(t[:, :1], mask[:, :1], 1)

    def test_shared_forward_matches_teacher_flow_and_combines_normalized_velocity(self):
        model = TinyPolicy()
        actions = torch.randn(2, 8, 20)
        mask = torch.ones_like(actions); mask[..., 17:] = 0
        inputs = {'inputs': {'action': actions, 'action_mask': mask}}
        for alternate in (False, True):
            model.action_head.config.use_alternate_vl_dit = alternate
            captured = {}
            handle = model.action_head.register_forward_hook(lambda module, args, output: captured.update(output))
            objective = ModelSLoss(ModelSConfig(velocity_reference=2, velocity_lambda=.3,
                                               gradient_diagnostics=True), torch.ones(7), torch.zeros(7))
            try:
                total, metrics = objective.forward(model, inputs, diagnostics=True)
            finally:
                handle.remove()
            expected = ((captured['velocity'].float()-captured['teacher_velocity'].float()).square()*mask).sum()/(mask.sum()+1e-6)
            torch.testing.assert_close(captured['loss'], expected)
            self.assertFalse(captured['teacher_velocity'].requires_grad)
            teacher, student = model.action_head.model.calls[-2:]
            torch.testing.assert_close(teacher[0], student[0])
            torch.testing.assert_close(teacher[1], student[1])
            self.assertEqual(teacher[2:], (True, False, False))
            self.assertEqual(student[2:], (False, True, False))
            self.assertFalse(model.action_head.model.adapter_disabled)
            self.assertTrue(model.action_head.model.training)
            self.assertAlmostEqual(metrics['normalized_velocity_loss'], metrics['raw_velocity_loss']/4, places=5)
            self.assertAlmostEqual(total.item(), metrics['fm_loss']+.3*metrics['normalized_velocity_loss'], delta=1e-3)
            self.assertGreater(metrics['fm_grad_norm'], 0)
            self.assertGreater(metrics['velocity_grad_norm'], 0)
            total.backward()
            self.assertIsNone(model.action_head.action_decoder.weight.grad)
            self.assertTrue(torch.isfinite(model.action_head.lora.grad))
            self.assertNotIn('forward', model.action_head.__dict__)

    def test_lambda_zero_is_exact_fm_and_changes_gradient_for_positive_lambda(self):
        model = TinyPolicy()
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        rng = torch.get_rng_state(); grads = []
        for weight in (0., .1):
            torch.set_rng_state(rng)
            objective = ModelSLoss(ModelSConfig(velocity_reference=1, velocity_lambda=weight), torch.ones(7), torch.zeros(7))
            total, metrics = objective.forward(model, inputs)
            if weight == 0:
                self.assertEqual(total.item(), metrics['fm_loss'])
                self.assertEqual(metrics['weighted_velocity_loss'], 0)
            grads.append(torch.autograd.grad(total, model.action_head.lora)[0])
        self.assertFalse(torch.allclose(grads[0], grads[1]))

    def test_target_reference_freezes_with_valid_pair_weighting_and_resumes(self):
        config = ModelSConfig(dt=1, calibration_batches=2)
        first = ModelSLoss(config, torch.ones(7), torch.zeros(7))
        t = torch.arange(4.).reshape(1, 4, 1).expand(1, 4, 20)
        first.calibrate({'inputs': {'action': t, 'action_mask': torch.ones_like(t)}})
        resumed = ModelSLoss(config, torch.ones(7), torch.zeros(7))
        resumed.load_state_dict(first.state_dict())
        mask = torch.ones_like(t); mask[:, 2:] = 0
        for objective in (first, resumed):
            objective.calibrate({'inputs': {'action': 3*t, 'action_mask': mask}})
            # 3 valid time pairs at speed 1, 1 pair at speed 3, across 7 joints.
            self.assertAlmostEqual(objective.reference**2, 3.)
            state = objective.state_dict()
            objective.calibrate({})
            self.assertEqual(objective.state_dict(), state)
        self.assertEqual(first.state_dict(), resumed.state_dict())
        with self.assertRaisesRegex(ValueError, 'new flow-preservation'):
            resumed.load_state_dict({'format': 'model-s-physical-jerk-only-v1'})
        changed = ModelSLoss(ModelSConfig(velocity_reference=2), torch.ones(7), torch.zeros(7))
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            changed.load_state_dict(first.state_dict())

    def test_stationary_reference_floor_and_config_validation(self):
        objective = ModelSLoss(ModelSConfig(calibration_batches=1), torch.ones(7), torch.zeros(7))
        action = torch.zeros(1, 3, 20)
        objective.calibrate({'inputs': {'action': action, 'action_mask': torch.ones_like(action)}})
        self.assertEqual(objective.reference, .001)
        for kwargs in ({'velocity_lambda': -1}, {'velocity_lambda': float('nan')},
                       {'velocity_reference': 0}, {'dt': 0}, {'calibration_batches': 0}):
            with self.assertRaises(ValueError):
                ModelSConfig(**kwargs)

    def test_training_loop_calibrates_without_optimizer_steps_and_logs_single_line(self):
        import train as entry
        from src import training_runtime as runtime
        from io import StringIO
        from contextlib import redirect_stdout
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(dt=1, calibration_batches=2), torch.ones(7), torch.zeros(7))
        args = SimpleNamespace(max_steps=2, gradient_accumulation=2, save_steps=2, no_hf_upload=True)
        inputs = {'inputs': {'action': torch.randn(1, 8, 20), 'action_mask': torch.ones(1, 8, 20)}}
        session = SimpleNamespace(model=model, step=0, smoothness=objective,
            batches=lambda count: [inputs]*count, autocast=nullcontext,
            trainer=SimpleNamespace(state=SimpleNamespace(log_history=[])))
        @contextmanager
        def setup(*args): yield session
        log = StringIO()
        with patch.object(entry, 'prepare_training', return_value=(None, {})), \
             patch.object(runtime, 'setup_training', setup), \
             patch.object(runtime, 'preprocess', side_effect=lambda batch, _: batch), \
             patch.object(runtime, 'update_weights') as update, \
             patch.object(runtime, 'save_checkpoint'), patch.object(runtime, 'export_model'), \
             patch.object(runtime, 'save_training_loss_plot'), \
             patch.object(runtime, 'save_training_distribution_plot'), redirect_stdout(log):
            entry.main(args)
        self.assertEqual(update.call_count, 2)
        self.assertEqual(objective.calibration_count, 2)
        self.assertEqual(sum(line.startswith('step=') for line in log.getvalue().splitlines()), 2)
        lines = [line for line in log.getvalue().splitlines() if line.startswith('step=')]
        for line, row in zip(lines, session.trainer.state.log_history):
            self.assertIn(f"fm_loss={row['fm_loss']:.6f}", line)
            self.assertIn(f"velocity_loss={row['normalized_velocity_loss']:.6f}", line)
            self.assertIn(f"total_loss={row['loss']:.6f}", line)
        for row in session.trainer.state.log_history:
            self.assertAlmostEqual(row['loss'], row['fm_loss']+row['weighted_velocity_loss'], places=5)
            self.assertNotIn('jerk_loss', row)
