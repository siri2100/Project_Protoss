from collections import OrderedDict
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from peft import PeftModel
import torch
from torch import nn

from src.groot_lora import install_lora, merge_lora


def make_test_dit():
    def block():
        return nn.Sequential(OrderedDict(attn1=nn.Sequential(OrderedDict(
            to_q=nn.Linear(4, 4), to_k=nn.Linear(4, 4), to_v=nn.Linear(4, 4),
            to_out=nn.Linear(4, 4),
        ))))
    return nn.Sequential(OrderedDict(transformer_blocks=nn.Sequential(*(block() for _ in range(4)))))


class LoRATests(unittest.TestCase):
    def test_initial_equivalence_only_adapter_gradients_and_merge(self):
        torch.manual_seed(42)
        model = nn.Module()
        model.backbone = nn.Linear(4, 4)
        model.action_head = nn.Module()
        model.action_head.model = make_test_dit()
        x = torch.randn(5, 4)
        original = model.action_head.model(x).detach()
        base_dit = copy.deepcopy(model.action_head.model)
        targets = install_lora(model, 2, 4)
        self.assertIsInstance(model.action_head.model, PeftModel)
        self.assertEqual(targets, [f"action_head.model.transformer_blocks.{i}.attn1.{projection}"
                                   for i in (1, 3) for projection in ("to_q", "to_k", "to_v")])
        torch.testing.assert_close(original, model.action_head.model(x))
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            model.action_head.model(x).square().mean().backward()
            optimizer.step()
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all((".lora_A." in name or ".lora_B." in name) for name in trainable))
        self.assertIsNone(model.backbone.weight.grad)
        adapted = model.action_head.model(x).detach()
        self.assertFalse(torch.equal(original, adapted))
        with tempfile.TemporaryDirectory() as tmp:
            model.action_head.model.save_pretrained(tmp)
            reloaded_adapter = PeftModel.from_pretrained(base_dit, tmp)
            torch.testing.assert_close(adapted, reloaded_adapter(x))
        merge_lora(model)
        torch.testing.assert_close(adapted, model.action_head.model(x))
        self.assertFalse(any("lora_" in key or "base_layer" in key for key in model.state_dict()))
        reloaded = make_test_dit()
        reloaded.load_state_dict(model.action_head.model.state_dict())
        torch.testing.assert_close(adapted, reloaded(x))

    def test_real_alternate_dit_forward_and_merge(self):
        # Load the actual GR00T DiT without importing its GPU/backbone dependencies.
        path = Path(__file__).resolve().parents[3] / "Issac-GR00T-N17/gr00t/model/modules/dit.py"
        spec = importlib.util.spec_from_file_location("protoss_test_dit", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dit = module.AlternateVLDiT(
            num_attention_heads=2, attention_head_dim=4, output_dim=8, num_layers=2,
            dropout=0.0, final_dropout=False, positional_embeddings=None,
            cross_attention_dim=8, interleave_self_attention=True,
        )
        model = nn.Module()
        model.action_head = nn.Module()
        model.action_head.model = dit
        model.eval()
        inputs = dict(hidden_states=torch.randn(2, 3, 8),
                      encoder_hidden_states=torch.randn(2, 4, 8),
                      timestep=torch.tensor([1, 2]),
                      image_mask=torch.tensor([[True, True, False, False]] * 2),
                      backbone_attention_mask=torch.ones(2, 4, dtype=torch.bool))
        expected = dit(**inputs).detach()
        targets = install_lora(model, 2, 4)
        self.assertEqual(targets, [f"action_head.model.transformer_blocks.1.attn1.{projection}"
                                   for projection in ("to_q", "to_k", "to_v")])
        model.eval()
        torch.testing.assert_close(model.action_head.model(**inputs), expected)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, ".lora_A." in name or ".lora_B." in name)
        model.action_head.model(**inputs).square().mean().backward()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
        optimizer.step()
        adapted = model.action_head.model(**inputs).detach()
        self.assertFalse(torch.equal(adapted, expected))
        merge_lora(model)
        torch.testing.assert_close(model.action_head.model(**inputs), adapted)
        self.assertIsInstance(model.action_head.model, module.AlternateVLDiT)


if __name__ == "__main__":
    unittest.main()
