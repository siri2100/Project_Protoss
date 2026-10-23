import argparse
import json
from pathlib import Path
import tempfile
import unittest
from src.analyze_groot import compare_processors, differences, trend


class AnalysisTests(unittest.TestCase):
    def test_processor_nested_stats_and_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mean in [('base', 0), ('fm/processor', 1)]:
                p = root / name
                p.mkdir(parents=True)
                (p / 'processor_config.json').write_text('{"delta_indices": [-15, 0]}')
                (p / 'statistics.json').write_text(json.dumps({'droid': {'mean': [mean]}}))
            compare_processors(argparse.Namespace(base_model=root/'base', model_path=root/'fm', output_dir=root/'out'))
            report = json.loads((root/'out/processor_diff.json').read_text())
            self.assertEqual(report['files']['processor_config.json'], [])
            self.assertEqual(report['files']['statistics.json'][0]['path'], '/droid/mean')
            self.assertTrue(differences({'a': 1}, {}))

    def test_trend_complete_logs_and_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root/'train/checkpoint-500'
            training.mkdir(parents=True)
            (training/'trainer_state.json').write_text('{"log_history": [{"step":500,"loss":0.1}]}')
            run = root/'eval/checkpoint-500'
            run.mkdir(parents=True)
            config = {'model':'FMD', 'model_path':'/exports/checkpoint-500/inference',
                      'seeds':[42], 'traj_ids':[0,1], 'dataset_manifest_sha256':'same',
                      'execution_horizon':8, 'steps':1000, 'denoising_steps':4}
            (run/'evaluation.json').write_text(json.dumps(config))
            (run/'seed_42.log').write_text('INFO:root:MSE for trajectory 0: 0.1, MAE: 0.2\nINFO:root:MSE for trajectory 1: 0.3, MAE: 0.4\nINFO:root:Done\n')
            args = argparse.Namespace(results_root=root/'eval', training_root=root/'train', output_dir=root/'out')
            trend(args)
            rows = json.loads((root/'out/trend.json').read_text())
            self.assertEqual(rows[0]['step'],500)
            self.assertAlmostEqual(rows[0]['mae'],0.3)
            self.assertTrue((root/'out/checkpoint-500_log_history.json').exists())
            other = root/'eval/other'
            other.mkdir()
            config['execution_horizon']=4
            (other/'evaluation.json').write_text(json.dumps(config))
            (other/'seed_42.log').write_text((run/'seed_42.log').read_text())
            with self.assertRaisesRegex(ValueError,'conditions differ'):
                trend(args)
            (other/'seed_42.log').write_text('INFO:root:Done')
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                trend(args)
