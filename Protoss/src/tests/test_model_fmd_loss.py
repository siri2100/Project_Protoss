"""Check pure flow distillation numerics, teacher freezing and PEFT initialization."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.model_fmd_loss import ModelFMDLoss
from src.tests.test_model_s_loss import TinyPolicy


class LinearDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(20, 20, bias=False)
        self.dropout = nn.Dropout(.7)

    def forward(self, hidden_states, **kwargs):
        return self.proj(self.dropout(hidden_states)), None


class ModelFMDLossTests(unittest.TestCase):
    def inputs(self):
        return {'inputs': {'action': torch.randn(2, 8, 20), 'action_mask': torch.ones(2, 8, 20)}}

    def test_loss_is_masked_teacher_student_mse_with_lora_gradient(self):
        torch.manual_seed(24)
        model, inputs = TinyPolicy(), self.inputs()
        inputs['inputs']['action_mask'][:, 4:, 17:] = 0
        outputs = []
        handle = model.action_head.register_forward_hook(lambda module, args, output: outputs.append(output))
        try:
            total, metrics = ModelFMDLoss().forward(model, inputs)
        finally:
            handle.remove()
        out = outputs[0]
        expected = (((out['velocity'] - out['teacher_velocity']).float().square() * out['action_mask']).sum()
                    / (out['action_mask'].sum() + 1e-6))
        torch.testing.assert_close(total, expected)
        self.assertEqual(metrics, {'fmd_loss': total.item()})
        self.assertFalse(out['teacher_velocity'].requires_grad)
        self.assertEqual(len(model.action_head.model.calls), 2)
        teacher, student = model.action_head.model.calls
        torch.testing.assert_close(teacher[0], student[0])
        torch.testing.assert_close(teacher[1], student[1])
        total.backward()
        self.assertGreater(model.action_head.lora.grad.abs().item(), 0)
        self.assertIsNone(model.action_head.action_decoder.weight.grad)
        self.assertNotIn('forward', model.action_head.__dict__)

    def test_default_peft_is_zero_and_distillation_only_restores_a_perturbed_student(self):
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(25)
        model, inputs = TinyPolicy(), self.inputs()
        model.action_head.model = get_peft_model(LinearDiT(), LoraConfig(
            r=2, lora_alpha=2, target_modules=['proj'], lora_dropout=0.))
        objective = ModelFMDLoss()
        initial, _ = objective.forward(model, inputs)
        self.assertEqual(initial.item(), 0.)
        initial.backward()
        projection = model.action_head.model.base_model.model.proj
        self.assertEqual(projection.lora_B['default'].weight.grad.abs().sum().item(), 0.)
        self.assertIsNone(projection.base_layer.weight.grad)
        self.assertTrue(model.action_head.model.base_model.model.dropout.training)
        with torch.no_grad():
            projection.lora_B['default'].weight.fill_(.1)
        model.zero_grad()
        rng = torch.get_rng_state()
        total, _ = objective.forward(model, inputs)
        self.assertGreater(total.item(), 0)
        total.backward()
        self.assertGreater(projection.lora_B['default'].weight.grad.abs().sum().item(), 0)
        optimizer = torch.optim.SGD([projection.lora_B['default'].weight], lr=.05)
        optimizer.step()
        torch.set_rng_state(rng)
        reduced, _ = objective.forward(model, inputs)
        self.assertLess(reduced.item(), total.item())
        self.assertIsNone(projection.base_layer.weight.grad)

    def test_failure_restores_native_head(self):
        model = TinyPolicy()
        with patch.object(model.action_head.model, 'forward', side_effect=RuntimeError('teacher failed')):
            with self.assertRaisesRegex(RuntimeError, 'teacher failed'):
                ModelFMDLoss().forward(model, self.inputs())
        self.assertNotIn('forward', model.action_head.__dict__)
        self.assertFalse(model.action_head.model.adapter_disabled)
        self.assertTrue(model.action_head.model.training)

    def test_checkpoint_identity_rejects_old_fm(self):
        loss = ModelFMDLoss()
        loss.load_state_dict(loss.state_dict())
        with self.assertRaisesRegex(ValueError, 'old FM'):
            loss.load_state_dict({'format': 'FM'})


if __name__ == '__main__':
    unittest.main()
