"""CPU checks for selecting the requested PEFT checkpoint without final-weight fallback."""
import json
from pathlib import Path
import tempfile
import unittest

from src.inference_loading import checkpoint_inference_args, resolve_inference_paths


class CheckpointInferenceTests(unittest.TestCase):
    def make_run(self, root):
        (root / 'lora_settings.json').write_text(json.dumps({'backend': 'peft', 'base_model': '/old/base'}))
        for folder in (root, root / 'checkpoint-500'):
            (folder / 'adapter').mkdir(parents=True)
            (folder / 'adapter/adapter_config.json').write_text('{}')
            (folder / 'adapter/adapter_model.safetensors').write_bytes(b'weights')
        (root / 'processor').mkdir()
        (root / 'processor/processor_config.json').write_text('{}')
        return root / 'checkpoint-500'

    def test_selects_step_adapter_and_parent_processor_with_base_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.make_run(root)
            base, adapter, processor = resolve_inference_paths(checkpoint, '/new/base')
            self.assertEqual(base, '/new/base')
            self.assertEqual(adapter, checkpoint / 'adapter')
            self.assertEqual(processor, root / 'processor')
            args = checkpoint_inference_args(['--model-path=' + str(root), '--port', '5557'], 500)
            self.assertEqual(args, ['--model-path', str(checkpoint), '--port', '5557'])

    def test_uses_checkpoint_processor_in_both_supported_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.make_run(root)
            (checkpoint / 'processor_config.json').write_text('{}')
            self.assertEqual(resolve_inference_paths(checkpoint)[2], checkpoint)
            (checkpoint / 'processor').mkdir()
            (checkpoint / 'processor/processor_config.json').write_text('{}')
            self.assertEqual(resolve_inference_paths(checkpoint)[2], checkpoint / 'processor')

    def test_missing_checkpoint_or_weights_never_use_final_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.make_run(root)
            with self.assertRaisesRegex(FileNotFoundError, 'Requested checkpoint is missing'):
                checkpoint_inference_args(['--model-path', str(root)], 1000)
            (checkpoint / 'adapter/adapter_model.safetensors').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'weights missing'):
                resolve_inference_paths(checkpoint)
            with self.assertRaisesRegex(ValueError, 'positive'):
                checkpoint_inference_args(['--model-path', str(root)], 0)

    def test_standalone_checkpoint_requires_explicit_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self.make_run(root)
            (root / 'lora_settings.json').unlink()
            with self.assertRaisesRegex(ValueError, 'base-model-path'):
                resolve_inference_paths(checkpoint)
            self.assertEqual(resolve_inference_paths(checkpoint, '/base')[0], '/base')
