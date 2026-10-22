"""FK golden poses from USD/Gf and recovery of absolute relative-action targets."""

from types import SimpleNamespace
import unittest

import torch

from src.droid_kinematics import DroidForwardKinematics
from src.joint_actions import JointActionDecoder


class DroidKinematicsTests(unittest.TestCase):
    def test_fk_preserves_float32_geometry_inside_training_autocast(self):
        q = torch.tensor([[.2, -.5, .3, -1.4, .1, 1.5, -.2]], requires_grad=True)
        fk = DroidForwardKinematics()
        expected = fk(q)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = fk(q)
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_positions_match_independent_usd_gf_composition(self):
        # Independently evaluated with Gf matrices on the pinned USD (meters).
        q = torch.tensor([[0.]*7, [0., -torch.pi/5, 0., -4*torch.pi/5, 0., 3*torch.pi/5, 0.],
                          [.2, -.5, .3, -1.4, .1, 1.5, -.2]], dtype=torch.float64)
        expected = torch.tensor([[.08799996452419188, 1.7338314102408707e-9, .9078258275985677],
                                 [.35970316407690267, 1.7338312474545307e-9, .47176719456267197],
                                 [.24815044645958922, .2279468813080661, .8865187169123363]], dtype=torch.float64)
        fk = DroidForwardKinematics()
        torch.testing.assert_close(fk(q), expected, atol=1e-10, rtol=0)
        self.assertTrue(torch.autograd.gradcheck(fk, (q.requires_grad_(),), atol=1e-6))

    def test_relative_joint_reference_is_restored_before_nonlinear_fk(self):
        decoder = JointActionDecoder(torch.ones(7), torch.zeros(7), relative=True)
        normalized = torch.zeros(2, 4, 7)
        normalized[:, :, 1] = torch.tensor([0., .1, .2, .3])
        reference = torch.tensor([[1.]*7, [2.]*7])
        # A normalized state may be clipped or dropped; it must not define FK.
        inputs = {'inputs': {'state': torch.zeros(2, 1, 10), 'raw_joint_reference': reference}}
        absolute = decoder.absolute(normalized, torch.ones_like(normalized), inputs)
        torch.testing.assert_close(absolute[0], normalized[0] + 1)
        torch.testing.assert_close(absolute[1], normalized[1] + 2)
        positions = DroidForwardKinematics()(absolute)
        lengths = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1).sum(-1)
        self.assertGreater((lengths[0] - lengths[1]).abs().item(), 1e-4)

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
