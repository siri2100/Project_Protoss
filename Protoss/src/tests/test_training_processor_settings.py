import json
from pathlib import Path
import tempfile
import unittest
from src.training_processor_settings import load_training_modalities

class TrainingProcessorTests(unittest.TestCase):
    def test_single_frame_retains_other_base_modality_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'processor_kwargs': {'modality_configs': {'droid': {
                'video': {'delta_indices': [-15,0], 'modality_keys': ['camera']},
                'state': {'delta_indices': [0]}, 'action': {'delta_indices': list(range(40))}}}}}
            (root/'processor_config.json').write_text(json.dumps(config))
            result = load_training_modalities(root,'droid')
            self.assertEqual(result['droid']['video']['delta_indices'],[0])
            self.assertEqual(result['droid']['action'],config['processor_kwargs']['modality_configs']['droid']['action'])
            self.assertEqual(json.loads((root/'processor_config.json').read_text()),config)
            with self.assertRaises(ValueError):load_training_modalities(root,'missing')
