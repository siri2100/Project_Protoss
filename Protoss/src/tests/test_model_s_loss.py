"""CPU checks for the scientific objective, same-forward capture and resume."""
from contextlib import contextmanager, nullcontext
from enum import Enum
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.model_s_loss import (
    ModelSConfig, ModelSLoss, capture_flow_forward, clean_action_endpoint,
    joint_affine, masked_jerk_loss, unnormalize_joints,
)


class TinyEncoder(nn.Module):
    def forward(self, noisy, t, embodiment):
        return noisy


class TinyHead(nn.Module):
    """Mirror upstream's interpolation, encoder/decoder calls and native FM mask."""
    def __init__(self):
        super().__init__()
        self.action_encoder = TinyEncoder()
        self.action_decoder = nn.Linear(20, 20, bias=False)
        self.action_decoder.requires_grad_(False)
        self.lora = nn.Parameter(torch.tensor(0.1))

    def sample_time(self, batch, **kwargs):
        return torch.full((batch,), 0.237, **kwargs)

    def forward(self, action, mask):
        noise = torch.randn_like(action)
        t = self.sample_time(action.shape[0], device=action.device, dtype=action.dtype)
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * action
        hidden = self.action_encoder(noisy, (t * 1000).long(), None) * self.lora
        # Include a state token to check velocity slicing.
        hidden = torch.cat([hidden[:, :1], hidden], dim=1)
        velocity = self.action_decoder(hidden)[:, -action.shape[1]:]
        loss = ((velocity - (action - noise)).square() * mask).sum() / (mask.sum() + 1e-6)
        return {"loss": loss, "action_mask": mask}


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_head = TinyHead()

    def forward(self, inputs):
        return self.action_head(inputs["action"], inputs["mask"])


class ModelSLossTests(unittest.TestCase):
    def test_training_loop_calibrates_then_uses_jerk_and_logs_components(self):
        import train as entry
        from src import training_runtime as runtime
        model = TinyPolicy()
        args = SimpleNamespace(max_steps=2, gradient_accumulation=2, save_steps=2,
                               no_hf_upload=True)
        objective = ModelSLoss(ModelSConfig(calibration_steps=1, dt=1), torch.ones(7), torch.zeros(7))
        inputs = {"inputs": {"action": torch.randn(1, 8, 20), "mask": torch.ones(1, 8, 20)}}
        session = SimpleNamespace(model=model, step=0, smoothness=objective,
                                  batches=lambda count: [inputs] * count,
                                  autocast=nullcontext,
                                  trainer=SimpleNamespace(state=SimpleNamespace(log_history=[])))

        @contextmanager
        def setup(*args):
            yield session

        with patch.object(entry, "prepare_training", return_value=(None, {})), \
             patch.object(runtime, "setup_training", setup), \
             patch.object(runtime, "preprocess", side_effect=lambda batch, _: batch), \
             patch.object(runtime, "update_weights"), \
             patch.object(runtime, "save_checkpoint") as save, \
             patch.object(runtime, "export_model"), \
             patch.object(runtime, "save_training_loss_plot"), \
             patch.object(runtime, "save_training_distribution_plot"):
            entry.main(args)
        rows = session.trainer.state.log_history
        self.assertEqual(rows[0]["jerk_weight"], 0)
        self.assertGreater(rows[1]["jerk_weight"], 0)
        for row in rows:
            self.assertAlmostEqual(row["loss"], row["fm_loss"] + row["weighted_jerk_loss"], places=5)
        save.assert_called_once_with(session)

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

    def test_same_forward_native_fm_preserved_and_jerk_reaches_lora(self):
        torch.manual_seed(4)
        model = TinyPolicy()
        inputs = {"action": torch.randn(2, 8, 20), "mask": torch.ones(2, 8, 20)}
        inputs["mask"][..., 17:] = 0
        objective = ModelSLoss(ModelSConfig(jerk_weight=.1, dt=1, gradient_diagnostics=True), torch.ones(7), torch.zeros(7))
        rng = torch.get_rng_state()
        native = model(inputs)["loss"]
        torch.set_rng_state(rng)
        total, metrics = objective.forward(model, {"inputs": inputs}, diagnostics=True)
        self.assertAlmostEqual(native.item(), metrics["fm_loss"])
        self.assertAlmostEqual(total.item(), metrics["fm_loss"] + .1 * metrics["jerk_loss"], places=5)
        self.assertGreater(metrics["jerk_grad_norm"], 0)
        total.backward()
        self.assertTrue(torch.isfinite(model.action_head.lora.grad))
        self.assertIsNone(model.action_head.action_decoder.weight.grad)
        self.assertNotIn("sample_time", model.action_head.__dict__)
        self.assertFalse(model.action_head.action_encoder._forward_pre_hooks)
        self.assertFalse(model.action_head.action_decoder._forward_hooks)

    def test_excludes_gripper_eef_and_padded_dimensions(self):
        torch.manual_seed(8)
        model = TinyPolicy()
        objective = ModelSLoss(ModelSConfig(jerk_weight=1, dt=1), torch.ones(7), torch.zeros(7))
        inputs = {"action": torch.randn(1, 8, 20), "mask": torch.ones(1, 8, 20)}
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

    def test_calibration_freezes_and_resume_preserves_accumulators(self):
        config = ModelSConfig(calibration_steps=2, jerk_ratio=.01)
        loss = ModelSLoss(config, torch.ones(7), torch.zeros(7))
        loss.finish_step({"fm_loss": 2., "jerk_loss": 4.})
        self.assertIsNone(loss.weight)
        resumed = ModelSLoss(config, torch.ones(7), torch.zeros(7))
        resumed.load_state_dict(loss.state_dict())
        for objective in (loss, resumed):
            objective.finish_step({"fm_loss": 4., "jerk_loss": 8.})
            self.assertEqual(objective.weight, .005)
            objective.finish_step({"fm_loss": 100., "jerk_loss": 1.})
            self.assertEqual(objective.weight, .005)
        self.assertEqual(loss.state_dict(), resumed.state_dict())
        wrong = ModelSLoss(ModelSConfig(dt=.1), torch.ones(7), torch.zeros(7))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            wrong.load_state_dict(loss.state_dict())

    def test_capture_cleanup_on_forward_failure(self):
        head = TinyHead()
        with self.assertRaises(RuntimeError):
            with capture_flow_forward(head):
                raise RuntimeError("forward failed")
        self.assertNotIn("sample_time", head.__dict__)
        self.assertFalse(head.action_decoder._forward_hooks)

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


if __name__ == "__main__":
    unittest.main()
