import unittest

import torch
from torch import nn

from src.pretrained_fm_loss import measure


class LossTests(unittest.TestCase):
    def test_native_mask_weighting_no_gradients_and_batch_limit(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, inputs):
                self.assert_no_grad = not torch.is_grad_enabled()
                error = self.weight * inputs["error"]
                mask = inputs["mask"]
                return {"loss": error.sum() / (mask.sum() + 1e-6),
                        "action_loss": error, "action_mask": mask}

        model = Model()
        batches = [{"inputs": {"error": torch.tensor([[2., 0.]]), "mask": torch.tensor([[1., 0.]])}},
                   {"inputs": {"error": torch.tensor([[4., 4.]]), "mask": torch.tensor([[1., 1.]])}},
                   {"inputs": {"error": torch.tensor([[100.]]), "mask": torch.tensor([[1.]])}}]
        result, rows = measure(model, batches, batches=2, device="cpu", bf16=False)
        self.assertAlmostEqual(result["fm_loss"], 10 / (3 + 1e-6))
        self.assertAlmostEqual(result["mean_batch_loss"], 3, places=5)
        self.assertEqual((result["samples"], len(rows)), (2, 2))
        self.assertTrue(model.assert_no_grad)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(model.weight.item(), 1)

    def test_empty_data_and_invalid_outputs_fail(self):
        with self.assertRaisesRegex(ValueError, "no batches"):
            measure(None, [], batches=1, device="cpu", bf16=False)
        for value in (float("nan"), 0.):
            def model(**inputs):
                return {"loss": torch.tensor(value), "action_loss": torch.tensor(value),
                        "action_mask": torch.tensor([[value]])}
            with self.assertRaises(ValueError):
                measure(model, [{}], batches=1, device="cpu", bf16=False)


if __name__ == "__main__":
    unittest.main()
