"""Time compression, physical normalization, limits and LoRA gradient checks."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.joint_actions import JointActionDecoder, action_affine
from src.model_e_loss import (ModelEConfig, ModelELoss, compressed_teacher_target,
                              joint_motion_limits)
from src.tests.test_model_s_loss import TinyPolicy


class ModelELossTests(unittest.TestCase):
    def forward(self, student, teacher, config=None, mask=None, reference=None, **kwargs):
        model = TinyPolicy()
        mask = torch.ones_like(student) if mask is None else mask
        reference = torch.zeros(student.shape[0], 7) if reference is None else reference
        inputs = {'inputs': {'raw_joint_reference': reference}}
        fmd = torch.where(mask.bool(), student - teacher.detach(), 0.).square().sum() / mask.sum()
        output = {'noisy': torch.zeros_like(student), 't': torch.zeros(student.shape[0]),
                  'velocity': student, 'teacher_velocity': teacher, 'action_mask': mask, 'loss': fmd}
        objective = ModelELoss(config or ModelEConfig(), kwargs.pop('scale', torch.ones(7)),
                              kwargs.pop('offset', torch.zeros(7)), **kwargs)
        with patch.object(model, 'forward', return_value=output) as forward:
            result = objective.forward(model, inputs)
        self.assertNotIn('raw_joint_reference', forward.call_args.kwargs['inputs'])
        self.assertIn('raw_joint_reference', inputs['inputs'])
        return result

    def test_linear_interpolation_and_terminal_hold_keep_joint_gripper_timing(self):
        teacher = torch.zeros(1, 5, 8, requires_grad=True)
        with torch.no_grad():
            teacher[..., :7] = torch.arange(5.).reshape(1, 5, 1)
            teacher[..., 7] = torch.tensor([0., 0., 1., 1., 1.])
        target, active = compressed_teacher_target(teacher, torch.ones_like(teacher), 1.5)
        torch.testing.assert_close(target[0, :, 0], torch.tensor([0., 1.5, 3., 4., 4.]))
        torch.testing.assert_close(target[0, :, 7], torch.tensor([0., .5, 1., 1., 1.]))
        self.assertTrue(active.all())
        self.assertFalse(target.requires_grad)
        unchanged, _ = compressed_teacher_target(teacher, active, 1.)
        torch.testing.assert_close(unchanged, teacher)

    def test_gaps_padding_and_short_runs_are_not_joined(self):
        teacher = torch.tensor([0., 1., 2., float('nan'), 10., 11., 12., float('nan'), 30.]).reshape(1, 9, 1)
        mask = torch.isfinite(teacher)
        target, active = compressed_teacher_target(teacher, mask, 2.)
        torch.testing.assert_close(target.flatten(), torch.tensor([0., 2., 2., 0., 10., 12., 12., 0., 0.]))
        self.assertEqual(active.flatten().tolist(), [True, True, True, False, True, True, True, False, False])

    def test_matching_fast_target_has_zero_loss_and_teacher_has_no_gradients(self):
        teacher = torch.zeros(1, 5, 20, requires_grad=True)
        with torch.no_grad():
            teacher[..., 10:17] = torch.arange(5.).reshape(1, 5, 1)
            teacher[..., 9] = torch.tensor([0., 0., 1., 1., 1.])
        motion = torch.cat((teacher[..., 10:17], teacher[..., 9:10]), dim=-1)
        target, _ = compressed_teacher_target(motion, torch.ones_like(motion), 1.5)
        student = teacher.detach().clone()
        student[..., 10:17], student[..., 9:10] = target[..., :7], target[..., 7:]
        student.requires_grad_()
        total, metrics = self.forward(student, teacher, ModelEConfig(fm_weight=0, limit_weight=0, speedup=1.5))
        self.assertEqual(metrics['fast_loss'], 0.)
        total.backward()
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_retiming_uses_physical_horizon_stats_and_frozen_channel_scales(self):
        student = torch.zeros(1, 4, 20, requires_grad=True)
        teacher = torch.zeros_like(student, requires_grad=True)
        scale = torch.arange(1., 5.).reshape(4, 1).repeat(1, 7)
        offset = torch.tensor([0., 1., 3., 6.]).reshape(4, 1).repeat(1, 7)
        total, metrics = self.forward(student, teacher, ModelEConfig(fm_weight=0, limit_weight=0, speedup=2),
                                      scale=scale, offset=offset)
        self.assertAlmostEqual(metrics['joint_fast_loss'], 13/64)
        self.assertEqual(metrics['gripper_fast_loss'], 0.)
        total.backward()
        self.assertEqual(student.grad[..., :9].abs().sum().item(), 0.)
        self.assertEqual(student.grad[..., 17:].abs().sum().item(), 0.)
        self.assertGreater(student.grad[..., 10:17].abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)

    def test_relative_joint_reference_cancels_in_tracking_and_limits(self):
        teacher = torch.zeros(1, 4, 20)
        teacher[..., 10:17] = torch.arange(4.).reshape(1, 4, 1)
        config = ModelEConfig(speedup=1.5, dt=1., velocity_limit=2., acceleration_limit=2.)
        values = []
        for origin in (0., 100.):
            student = teacher.clone().requires_grad_()
            total, _ = self.forward(student, teacher, config, reference=torch.full((1, 7), origin),
                decoder=JointActionDecoder(torch.ones(7), torch.zeros(7), relative=True))
            values.append(total.detach())
        torch.testing.assert_close(*values)

    def test_limits_include_current_state_to_first_action_and_use_dimensionless_excess(self):
        config = ModelEConfig(dt=1., velocity_limit=1., acceleration_limit=1.)
        joints = torch.tensor([2., 2., 2.]).reshape(1, 3, 1).repeat(1, 1, 7).requires_grad_()
        velocity, acceleration = joint_motion_limits(joints, torch.ones_like(joints), torch.zeros(1, 7), config)
        torch.testing.assert_close(velocity, torch.tensor(1/3))
        torch.testing.assert_close(acceleration, torch.tensor(.5))
        (velocity + acceleration).backward()
        self.assertGreater(joints.grad.abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, 'raw_joint_reference'):
            joint_motion_limits(joints, torch.ones_like(joints), None, config)

    def test_limits_ignore_missing_intervals_and_have_zero_finite_stationary_gradient(self):
        config = ModelEConfig(dt=1., velocity_limit=2., acceleration_limit=10.)
        joints = torch.tensor([0., 1., float('nan'), 10., 11.]).reshape(1, 5, 1).repeat(1, 1, 7).requires_grad_()
        velocity, acceleration = joint_motion_limits(joints, torch.isfinite(joints), torch.zeros(1, 7), config)
        self.assertEqual((velocity.item(), acceleration.item()), (0., 0.))
        (velocity + acceleration).backward()
        self.assertTrue(torch.isfinite(joints.grad).all())
        stationary = torch.ones(1, 4, 7, requires_grad=True)
        losses = joint_motion_limits(stationary, torch.ones_like(stationary), torch.ones(1, 7), config)
        sum(losses).backward()
        self.assertEqual(stationary.grad.abs().sum().item(), 0.)

    def test_padded_nans_do_not_contaminate_losses_or_gradients(self):
        teacher = torch.randn(1, 8, 20)
        teacher[:, 3] = float('nan')
        teacher[:, 7] = float('nan')
        student = teacher.clone().requires_grad_()
        mask = torch.isfinite(teacher)
        total, _ = self.forward(student, teacher, mask=mask)
        total.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad[:, [3, 7]].abs().sum().item(), 0.)
        with self.assertRaisesRegex(ValueError, 'three-point'):
            self.forward(torch.zeros(1, 2, 20), torch.zeros(1, 2, 20))

    def test_each_weight_scales_its_loss_and_lora_gradient(self):
        torch.manual_seed(123)
        model = TinyPolicy()
        inputs = {'inputs': {'action': torch.randn(2, 8, 20), 'action_mask': torch.ones(2, 8, 20),
                             'raw_joint_reference': torch.zeros(2, 7)}}
        rng = torch.get_rng_state()
        for term in ('fm_weight', 'fast_weight', 'limit_weight'):
            losses, gradients = [], []
            for weight in (1., 10.):
                torch.set_rng_state(rng)
                settings = dict(fm_weight=0., fast_weight=0., limit_weight=0.)
                settings[term] = weight
                objective = ModelELoss(ModelEConfig(**settings), torch.ones(7), torch.zeros(7))
                total, metrics = objective.forward(model, inputs)
                losses.append(total.detach())
                gradients.append(torch.autograd.grad(total, model.action_head.lora)[0])
                self.assertAlmostEqual(total.item(), sum(metrics[k] for k in
                    ('weighted_fmd_loss', 'weighted_fast_loss', 'weighted_limit_loss')), places=4)
            self.assertGreater(gradients[0].abs().item(), 0.)
            torch.testing.assert_close(losses[1], 10 * losses[0])
            torch.testing.assert_close(gradients[1], 10 * gradients[0])
        self.assertNotIn('forward', model.action_head.__dict__)
        self.assertFalse(model.action_head.model.adapter_disabled)
        self.assertIsNone(model.action_head.action_decoder.weight.grad)

    def test_fast_term_can_start_learning_when_initial_fmd_is_zero(self):
        torch.manual_seed(123)
        model = TinyPolicy()
        model.action_head.model.lora.data.zero_()
        inputs = {'inputs': {'action': torch.randn(2, 8, 20), 'action_mask': torch.ones(2, 8, 20),
                             'raw_joint_reference': torch.zeros(2, 7)}}
        objective = ModelELoss(ModelEConfig(limit_weight=0), torch.ones(7), torch.zeros(7))
        total, metrics = objective.forward(model, inputs)
        self.assertEqual(metrics['fmd_loss'], 0.)
        self.assertGreater(metrics['fast_loss'], 0.)
        total.backward()
        self.assertGreater(model.action_head.lora.grad.abs().item(), 0.)

    def test_checkpoint_and_configuration_validation(self):
        objective = ModelELoss(ModelEConfig(fm_weight=1000, fast_weight=.3), torch.ones(7), torch.zeros(7))
        state = copy.deepcopy(objective.state_dict())
        objective.load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'configuration differs'):
            ModelELoss(ModelEConfig(), torch.ones(7), torch.zeros(7)).load_state_dict(state)
        for old in ('model-e-flow-preservation-ee-path-length-v1', 'model-e-flow-preservation-path-length-v1'):
            with self.assertRaisesRegex(ValueError, 'new training run'):
                objective.load_state_dict({'format': old})
        for name in ('fm_weight', 'fast_weight', 'limit_weight'):
            for value in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError): ModelEConfig(**{name: value})
        for name in ('dt', 'velocity_limit', 'acceleration_limit'):
            for value in (0., -1., float('nan')):
                with self.assertRaises(ValueError): ModelEConfig(**{name: value})
        for value in (.9, float('inf')):
            with self.assertRaises(ValueError): ModelEConfig(speedup=value)

    def test_gripper_normalization_uses_its_own_active_statistics(self):
        from types import SimpleNamespace
        modality = SimpleNamespace(mean_std_embedding_keys=[])
        sap = SimpleNamespace(modality_configs={'droid': {'action': modality}},
            norm_params={'droid': {'action': {'gripper_position': {'dim': 1, 'min': [[0.], [2.]], 'max': [[2.], [6.]]}}}})
        scale, offset = action_affine(SimpleNamespace(state_action_processor=sap), 'droid', 'gripper_position', 1)
        torch.testing.assert_close(scale, torch.tensor([[1.], [2.]]))
        torch.testing.assert_close(offset, torch.tensor([[1.], [4.]]))


    def test_decoder_uses_active_action_normalization_and_requires_raw_reference(self):
        cfg = SimpleNamespace(rep=SimpleNamespace(value='relative'), type=SimpleNamespace(value='non_eef'),
                              format=SimpleNamespace(value='default'), state_key='joint_position')
        modality = SimpleNamespace(modality_keys=['eef_9d','gripper_position','joint_position'],
                                   action_configs=[None,None,cfg], mean_std_embedding_keys=None)
        state_cfg = SimpleNamespace(modality_keys=['joint_position','eef_9d','gripper_position'],
                                    mean_std_embedding_keys=None, sin_cos_embedding_keys=None)
        groups = {'joint_position':{'dim':7,'min':[-2.]*7,'max':[4.]*7},
                  'eef_9d':{'dim':9},'gripper_position':{'dim':1}}
        sap = SimpleNamespace(modality_configs={'droid':{'action':modality,'state':state_cfg}},
            norm_params={'droid':{'action':groups,'state':groups}}, use_relative_action=True,
            apply_sincos_state_encoding=False)
        decoder = JointActionDecoder.from_processor(SimpleNamespace(state_action_processor=sap),'droid')
        self.assertTrue(decoder.relative)
        result = decoder.absolute(torch.zeros(1,2,7),torch.ones(1,2,7),
                                  {'inputs':{'state':torch.zeros(1,1,17),
                                             'raw_joint_reference':torch.full((1,7),3.)}})
        torch.testing.assert_close(result,torch.full((1,2,7),4.))
        with self.assertRaisesRegex(ValueError,'requires raw_joint_reference'):
            decoder.absolute(torch.zeros(1,2,7),torch.ones(1,2,7),{'inputs':{}})


if __name__ == '__main__':
    unittest.main()
