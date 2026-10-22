"""Numerical and gradient checks for FMD + FK end-effector path length."""

import copy
import unittest

import torch

from src.model_e_loss import ModelEConfig, ModelELoss, masked_path_length_loss
from src.tests.test_model_s_loss import TinyPolicy


class ModelELossTests(unittest.TestCase):
    def test_relative_reference_drives_fk_without_entering_model_features(self):
        from unittest.mock import patch
        from src.joint_actions import JointActionDecoder
        model = TinyPolicy()
        scale, offset = torch.ones(7), torch.zeros(7)
        objective = ModelELoss(ModelEConfig(fm_weight=0, path_length_weight=1), scale, offset,
                               decoder=JointActionDecoder(scale, offset, relative=True))
        clean = torch.zeros(1, 4, 20, requires_grad=True)
        with torch.no_grad():
            clean[0, :, 11] = torch.tensor([0., .1, .3, .6])
        reference = torch.tensor([[.2, -.5, .3, -1.4, .1, 1.5, -.2]])
        inputs = {'inputs': {'raw_joint_reference': reference, 'state': torch.zeros(1, 1, 17)}}
        output = {'noisy': clean, 't': torch.ones(1), 'velocity': clean * 0,
                  'action_mask': torch.ones_like(clean), 'loss': clean.sum() * 0}
        with patch.object(model, 'forward', return_value=output) as forward:
            total, _ = objective.forward(model, inputs)
        self.assertNotIn('raw_joint_reference', forward.call_args.kwargs['inputs'])
        self.assertIn('raw_joint_reference', inputs['inputs'])
        positions = objective.kinematics(clean[..., 10:17] + reference[:, None])
        torch.testing.assert_close(total, (positions[:, 1:] - positions[:, :-1]).norm(dim=-1).sum())
        total.backward()
        self.assertGreater(clean.grad[..., 10:17].abs().sum().item(), 0.)

    def test_path_is_sum_of_euclidean_segments_then_batch_mean(self):
        joints = torch.zeros(2, 3, 3)
        joints[0, 1:, 0] = 3
        joints[0, 2, 1] = 4  # two segments: 3 + 4 = 7, not straight-line 5
        joints[1] = joints[0] * 2
        loss = masked_path_length_loss(joints, torch.ones_like(joints))
        torch.testing.assert_close(loss, torch.tensor(10.5))
        diagonal = torch.zeros(1, 2, 3)
        diagonal[0, 1, :2] = torch.tensor([3., 4.])
        torch.testing.assert_close(masked_path_length_loss(diagonal, torch.ones_like(diagonal)), torch.tensor(5.))

    def test_masked_nan_gaps_and_empty_chunks_do_not_contaminate_gradients(self):
        joints = torch.zeros(2, 5, 3)
        joints[0, :, 0] = torch.tensor([0., float("nan"), 10., 13., float("nan")])
        joints[1] = float("nan")
        mask = torch.zeros_like(joints)
        mask[0, [0, 2, 3]] = 1
        joints.requires_grad_()
        loss = masked_path_length_loss(joints, mask)
        torch.testing.assert_close(loss, torch.tensor(3.))
        loss.backward()
        self.assertTrue(torch.isfinite(joints.grad).all())
        self.assertEqual(joints.grad[0, [0, 1, 4]].abs().sum().item(), 0.)
        self.assertEqual(joints.grad[1].abs().sum().item(), 0.)

    def test_stationary_path_has_zero_finite_gradient(self):
        joints = torch.ones(2, 4, 3, requires_grad=True)
        loss = masked_path_length_loss(joints, torch.ones_like(joints))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(torch.isfinite(joints.grad).all())
        self.assertEqual(joints.grad.abs().sum().item(), 0.)

    def test_partial_joint_vectors_are_excluded_and_invalid_batches_fail(self):
        joints = torch.arange(3.).reshape(1, 3, 1).expand(-1, -1, 3)
        mask = torch.ones_like(joints)
        mask[0, 1, 0] = 0
        with self.assertRaisesRegex(ValueError, "no valid adjacent"):
            masked_path_length_loss(joints, mask)
        with self.assertRaisesRegex(ValueError, "at least two"):
            masked_path_length_loss(joints[:, :1], mask[:, :1])

    def test_both_weights_scale_their_losses_and_lora_gradients(self):
        torch.manual_seed(123)
        model = TinyPolicy()
        inputs = {"inputs": {"action": torch.randn(2, 8, 20), "action_mask": torch.ones(2, 8, 20)}}
        rng = torch.get_rng_state()
        for term in ("fmd", "path"):
            values, gradients = [], []
            for weight in (1., 10.):
                torch.set_rng_state(rng)
                config = ModelEConfig(fm_weight=weight if term == "fmd" else 0,
                                      path_length_weight=weight if term == "path" else 0)
                objective = ModelELoss(config, torch.ones(7), torch.zeros(7))
                total, metrics = objective.forward(model, inputs)
                values.append(total.detach())
                gradients.append(torch.autograd.grad(total, model.action_head.lora)[0])
                self.assertAlmostEqual(total.item(), metrics["weighted_fmd_loss"] + metrics["weighted_path_length_loss"], places=4)
            self.assertGreater(gradients[0].abs().item(), 0.)
            torch.testing.assert_close(values[1], 10 * values[0])
            torch.testing.assert_close(gradients[1], 10 * gradients[0])
        self.assertNotIn("forward", model.action_head.__dict__)
        self.assertFalse(model.action_head.model.adapter_disabled)
        self.assertIsNone(model.action_head.action_decoder.weight.grad)

    def test_physical_path_uses_horizon_specific_affine_and_only_joints(self):
        from unittest.mock import patch
        model = TinyPolicy()
        clean = torch.zeros(1, 4, 20, requires_grad=True)
        # EEF/gripper can change by huge amounts without entering the path.
        with torch.no_grad():
            clean[0, :, :10] = torch.arange(4.).reshape(4, 1) * 1000
        offsets = torch.zeros(4, 7)
        offsets[:, 0] = torch.tensor([0., 1., 3., 6.])
        output = {"noisy": clean, "t": torch.ones(1), "velocity": clean * 0,
                  "action_mask": torch.ones_like(clean), "loss": clean.sum() * 0}
        objective = ModelELoss(ModelEConfig(fm_weight=0, path_length_weight=1), torch.ones(4, 7), offsets)
        with patch.object(model, "forward", return_value=output):
            total, metrics = objective.forward(model, {})
        expected = objective.kinematics(offsets[None])
        expected = (expected[:, 1:] - expected[:, :-1]).norm(dim=-1).sum()
        self.assertAlmostEqual(metrics["path_length_loss"], expected.item(), places=6)
        total.backward()
        self.assertEqual(clean.grad[..., :10].abs().sum().item(), 0.)
        self.assertGreater(clean.grad[..., 10:17].abs().sum().item(), 0.)

    def test_checkpoint_preserves_weights_and_rejects_old_or_changed_objective(self):
        objective = ModelELoss(ModelEConfig(fm_weight=1000, path_length_weight=.3), torch.ones(7), torch.zeros(7))
        state = copy.deepcopy(objective.state_dict())
        objective.load_state_dict(state)
        self.assertEqual(objective.state_dict(), state)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            ModelELoss(ModelEConfig(), torch.ones(7), torch.zeros(7)).load_state_dict(state)
        with self.assertRaisesRegex(ValueError, "training checkpoint"):
            objective.load_state_dict({"format": "FM"})

    def test_weights_must_be_finite_and_nonnegative(self):
        for key in ("fm_weight", "path_length_weight"):
            for value in (-1., float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    ModelEConfig(**{key: value})

    def test_masked_nan_joints_are_removed_before_fk(self):
        from unittest.mock import patch
        model = TinyPolicy()
        clean = torch.randn(1, 5, 20)
        clean[:, 2, 10:17] = float('nan')
        clean.requires_grad_()
        mask = torch.ones_like(clean)
        mask[:, 2] = 0
        output = {'noisy':clean, 't':torch.ones(1), 'velocity':torch.zeros_like(clean),
                  'action_mask':mask, 'loss':torch.where(mask.bool(),clean,0.).sum()*0}
        objective = ModelELoss(ModelEConfig(fm_weight=0,path_length_weight=1),torch.ones(7),torch.zeros(7))
        with patch.object(model,'forward',return_value=output):
            total,_ = objective.forward(model,{})
        total.backward()
        self.assertTrue(torch.isfinite(clean.grad).all())
        self.assertEqual(clean.grad[:, 2].abs().sum().item(),0.)

    def test_old_joint_path_and_different_geometry_cannot_resume(self):
        objective = ModelELoss(ModelEConfig(),torch.ones(7),torch.zeros(7))
        with self.assertRaisesRegex(ValueError,'joint-path'):
            objective.load_state_dict({'format':'model-e-flow-preservation-path-length-v1'})
        state=objective.state_dict()
        state['geometry_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'geometry differs'):
            objective.load_state_dict(state)


if __name__ == "__main__":
    unittest.main()
