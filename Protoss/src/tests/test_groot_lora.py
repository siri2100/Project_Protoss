import unittest
import torch
from torch import nn

from src.groot_lora import install_lora, merge_lora


class LoRATests(unittest.TestCase):
    def test_initial_equivalence_only_adapter_gradients_and_merge(self):
        torch.manual_seed(42)
        model = nn.Module()
        model.backbone = nn.Linear(4, 4)
        model.action_head = nn.Module()
        model.action_head.model = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 3))
        x = torch.randn(5, 4)
        original = model.action_head.model(x).detach()
        targets = install_lora(model, 2, 4)
        self.assertEqual(len(targets), 2)
        torch.testing.assert_close(original, model.action_head.model(x))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            model.action_head.model(x).square().mean().backward()
            optimizer.step()
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all(name.endswith((".A", ".B")) for name in trainable))
        self.assertIsNone(model.backbone.weight.grad)
        adapted = model.action_head.model(x).detach()
        self.assertFalse(torch.equal(original, adapted))
        merge_lora(model)
        torch.testing.assert_close(adapted, model.action_head.model(x))
        self.assertFalse(any("parametrizations" in key for key in model.state_dict()))
        reloaded = nn.Sequential(nn.Linear(4, 6), nn.SiLU(), nn.Linear(6, 3))
        reloaded.load_state_dict(model.action_head.model.state_dict())
        torch.testing.assert_close(adapted, reloaded(x))


if __name__ == "__main__":
    unittest.main()
