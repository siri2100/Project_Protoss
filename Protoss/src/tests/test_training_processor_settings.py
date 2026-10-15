import json
from pathlib import Path
import tempfile
import unittest
from src.training_processor_settings import load_training_modalities

class TrainingProcessorTests(unittest.TestCase):
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
        from src.training_processor_settings import create_training_dataset
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
            class Factory:
                def __init__(self, config):
                    pass
                def build(self, processor):
                    return ('train', None)
            factory.DatasetFactory = Factory
            with patch.dict(sys.modules, {'gr00t.data.dataset.factory': factory}), \
                 patch('transformers.AutoProcessor.from_pretrained', return_value=processor) as load:
                self.assertEqual(create_training_dataset(pipeline, root / 'out/experiment_cfg'), ('train', None))
                load.assert_called_once_with(base)
                self.assertEqual(processor.state_action_processor.statistics, stats)
                self.assertEqual((root / 'out/processor/statistics.json').read_bytes(), (base / 'statistics.json').read_bytes())
                def corrupt(self, processor):
                    processor.state_action_processor.statistics['droid']['action']['joint']['mean'] = [999.0]
                    return ('train', None)
                with patch.object(Factory, 'build', corrupt):
                    with self.assertRaisesRegex(ValueError, 'changed pretrained normalization'):
                        create_training_dataset(pipeline, root / 'out/experiment_cfg')
