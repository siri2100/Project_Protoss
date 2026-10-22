import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from src.training_processor_settings import load_training_modalities


class ClippingProcessor:
    """Mimic state clipping; the raw joint reference must survive unchanged."""
    def __init__(self):
        self.modality_configs = {'droid': {'action': SimpleNamespace(
            action_configs=[None, None, SimpleNamespace(state_key='joint_position')])}}

    def __call__(self, messages):
        return {'state': np.clip(messages[0]['content'].states['joint_position'], -1, 1)}

class TrainingProcessorTests(unittest.TestCase):
    def test_raw_reference_survives_clipping_pickling_and_real_collator(self):
        import ast
        import pickle
        import torch
        from transformers import BatchFeature
        from src.training_processor_settings import JointReferenceProcessor
        raw = np.arange(14, dtype=np.float32).reshape(2, 7)
        messages = [{'content': SimpleNamespace(embodiment=SimpleNamespace(value='droid'),
                                               states={'joint_position': raw})}]
        original = ClippingProcessor()
        wrapped = pickle.loads(pickle.dumps(JointReferenceProcessor(original)))
        features = wrapped(messages)
        np.testing.assert_array_equal(features['state'], original(messages)['state'])
        np.testing.assert_array_equal(features['raw_joint_reference'], raw[-1])
        raw[-1] = 100  # wrapper saved a copy, not a view that may be mutated later
        np.testing.assert_array_equal(features['raw_joint_reference'], np.arange(7, 14))
        # Exercise the pinned collator's implementation without loading a VLM.
        source = Path(__file__).resolve().parents[3] / 'Issac-GR00T-N17/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py'
        cls = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'Gr00tN1d7DataCollator')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__call__')
        namespace = {'torch': torch, 'np': np, 'BatchFeature': BatchFeature, 'Dict': dict, 'Any': object}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        batch = namespace['__call__'](None, [features, features])['inputs']
        torch.testing.assert_close(batch['raw_joint_reference'], torch.arange(7., 14.).repeat(2, 1))
        self.assertEqual(batch['state'].shape, (2, 2, 7))

    def test_preserves_original_video_history_and_all_modality_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'processor_kwargs': {'modality_configs': {'droid': {
                'video': {'delta_indices': [-15,0], 'modality_keys': ['camera']},
                'state': {'delta_indices': [0]}, 'action': {'delta_indices': list(range(40))}}}}}
            (root/'processor_config.json').write_text(json.dumps(config))
            result = load_training_modalities(root,'droid')
            self.assertEqual(result, config['processor_kwargs']['modality_configs'])
            self.assertEqual(result['droid']['action'],config['processor_kwargs']['modality_configs']['droid']['action'])
            self.assertEqual(json.loads((root/'processor_config.json').read_text()),config)
            with self.assertRaises(ValueError):load_training_modalities(root,'missing')

    def test_original_artifacts_are_copied_without_serialization_changes(self):
        from src.training_processor_settings import processor_contract, save_original_processor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base/processor'
            base.mkdir(parents=True)
            artifacts = {'processor_config.json': '{"original": true}\n',
                         'statistics.json': '{"droid": {"action": {"mean": [1.25]}}}\n',
                         'embodiment_id.json': '{"droid":24}\n'}
            for name, content in artifacts.items():
                (base / name).write_text(content)
            output = root / 'output'
            save_original_processor(base.parent, output)
            self.assertEqual(processor_contract(base.parent), processor_contract(output))
            for name, content in artifacts.items():
                self.assertEqual((output / name).read_bytes(), (base / name).read_bytes())

    def test_dataset_creation_retains_stats_and_rejects_any_mutation(self):
        from copy import deepcopy
        import sys
        import types
        from unittest.mock import patch
        from src.training_processor_settings import create_training_dataset, JointReferenceProcessor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base'
            base.mkdir()
            stats = {'droid': {'action': {'joint': {'mean': [1.0]}}}}
            (base / 'processor_config.json').write_text('{}')
            (base / 'statistics.json').write_text(json.dumps(stats))
            processor = types.SimpleNamespace(state_action_processor=types.SimpleNamespace(statistics=deepcopy(stats)))
            config = types.SimpleNamespace(
                training=types.SimpleNamespace(start_from_checkpoint=str(base)),
                data=types.SimpleNamespace(override_pretraining_statistics=False,
                     datasets=[types.SimpleNamespace(embodiment_tag='droid')]))
            pipeline = types.SimpleNamespace(config=config)
            factory = types.ModuleType('gr00t.data.dataset.factory')
            received = []
            class Factory:
                def __init__(self, config):
                    pass
                def build(self, processor):
                    received.append(processor)
                    return ('train', None)
            factory.DatasetFactory = Factory
            with patch.dict(sys.modules, {'gr00t.data.dataset.factory': factory}), \
                 patch('transformers.AutoProcessor.from_pretrained', return_value=processor) as load:
                self.assertEqual(create_training_dataset(pipeline, root / 'out/experiment_cfg'), ('train', None))
                load.assert_called_once_with(base)
                self.assertIs(received[-1], processor)
                create_training_dataset(pipeline, root / 'out/experiment_cfg', preserve_joint_reference=True)
                self.assertIsInstance(received[-1], JointReferenceProcessor)
                self.assertIs(received[-1]._processor, processor)
                self.assertIs(pipeline.processor, processor)
                self.assertEqual(processor.state_action_processor.statistics, stats)
                self.assertEqual((root / 'out/processor/statistics.json').read_bytes(), (base / 'statistics.json').read_bytes())
                def corrupt(self, processor):
                    processor.state_action_processor.statistics['droid']['action']['joint']['mean'] = [999.0]
                    return ('train', None)
                with patch.object(Factory, 'build', corrupt):
                    with self.assertRaisesRegex(ValueError, 'changed pretrained normalization'):
                        create_training_dataset(pipeline, root / 'out/experiment_cfg')
