"""Diagnose GR00T LoRA equivalence, processor differences and checkpoint trends."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def processor_path(root):
    if (root / 'processor_config.json').exists():
        return root
    if (root / 'processor/processor_config.json').exists():
        return root / 'processor'
    raise ValueError(f'Processor config missing: {root}')


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f'{prefix}/{key}'
            if key not in left or key not in right:
                result.append({'path': path, 'baseline': left.get(key), 'candidate': right.get(key)})
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    return [] if left == right else [{'path': prefix, 'baseline': left, 'candidate': right}]


def compare_processors(args):
    roots = [processor_path(p) for p in (args.base_model, args.model_path)]
    files = [set(p.relative_to(root).as_posix() for p in root.rglob('*.json')
                 if '.cache' not in p.parts and '.git' not in p.parts) for root in roots]
    # Compare processor artifacts, excluding model/trainer configs when at root.
    names = {n for n in files[0] | files[1] if any(x in n.lower() for x in
             ('processor', 'stat', 'normal', 'modality', 'embodiment'))}
    result = {}
    for name in sorted(names):
        pair = [json.loads((root / name).read_text()) if name in f else None
                for root, f in zip(roots, files)]
        result[name] = differences(*pair)
    write(args.output_dir / 'processor_diff.json', {'roots': list(map(str, roots)), 'files': result})
    print('Processor differences:', {n: len(v) for n, v in result.items()}, flush=True)


def equivalence(args):
    import copy
    import numpy as np
    import torch
    from transformers import set_seed
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from .dataset_paths import split_path
    from .groot_lora import install_lora, merge_lora
    policy = Gr00tPolicy(embodiment_tag=args.tag, model_path=str(args.base_model), device=args.device)
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    configs = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(split_path(args.dataset_root, 'test')), modality_configs=configs)
    obs_configs = copy.deepcopy(configs)
    obs_configs.pop('action')
    observations = []
    for traj_id in args.traj_ids:
        traj = loader[traj_id]
        for step in args.frame_ids:
            if step >= len(traj):
                raise ValueError(f'Frame {step} beyond trajectory {traj_id}')
            point = extract_step_data(traj, step, obs_configs, policy.embodiment_tag)
            obs = {f'state.{k}': v for k, v in point.states.items()}
            obs.update({f'video.{k}': np.array(v) for k, v in point.images.items()})
            obs.update({key: point.text for key in configs['language'].modality_keys})
            observations.append((traj_id, step, parse_observation_gr00t(obs, configs)))
    def predict():
        predictions = []
        for _, _, obs in observations:
            set_seed(args.seed)
            with torch.inference_mode():
                action, _ = policy.get_action(copy.deepcopy(obs))
            predictions.append({k: np.asarray(v).copy() for k, v in action.items()})
        return predictions
    baseline = predict()
    repeat = predict()
    targets = install_lora(policy.model, args.rank, args.alpha)
    policy.model.eval()
    adapter = predict()
    merge_lora(policy.model)
    merged = predict()
    # Also test disk export + actual native reload, not just in-memory merge.
    export = args.output_dir / 'zero_lora_inference'
    if export.exists():
        raise ValueError(f'Export exists: {export}; use a new output directory')
    policy.model.save_pretrained(export)
    policy.processor.save_pretrained(export)
    del policy
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    policy = Gr00tPolicy(embodiment_tag=args.tag, model_path=str(export), device=args.device)
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    reloaded = predict()
    results = []
    for label, values in [('baseline_repeat', repeat), ('zero_lora', adapter), ('merged', merged), ('reloaded', reloaded)]:
        for (traj_id, step, _), original, current in zip(observations, baseline, values):
            if original.keys() != current.keys():
                raise ValueError('Action keys changed')
            for key in original:
                if original[key].shape != current[key].shape:
                    raise ValueError(f'Action shape changed: {key}')
                results.append({'stage': label, 'trajectory': traj_id, 'frame': step, 'action': key,
                    'max_abs_diff': float(np.max(np.abs(original[key] - current[key]))),
                    'allclose': bool(np.allclose(original[key], current[key], atol=args.atol, rtol=args.rtol))})
    passed = all(r['allclose'] for r in results)
    write(args.output_dir / 'equivalence.json', {'passed': passed, 'seed': args.seed,
        'atol': args.atol, 'rtol': args.rtol, 'targets': targets, 'comparisons': results})
    print(f'Equivalence passed: {passed}', flush=True)
    if not passed:
        raise RuntimeError('Equivalence failed; inspect baseline_repeat first for nondeterminism')


def export_checkpoint(args):
    import torch
    import gr00t.model  # register AutoModel/AutoProcessor
    from transformers import AutoModel, AutoProcessor
    from safetensors.torch import load_file
    from .groot_lora import install_lora, merge_lora
    checkpoint = args.checkpoint.resolve()
    settings = json.loads((checkpoint.parent / 'lora_settings.json').read_text())
    if settings.get('backend') != 'peft':
        raise ValueError('Legacy custom-LoRA checkpoint: export with the previous implementation')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output must be empty')
    processor = processor_path(checkpoint) if ((checkpoint / 'processor_config.json').exists() or
                (checkpoint / 'processor/processor_config.json').exists()) else processor_path(checkpoint.parent)
    model = AutoModel.from_pretrained(args.base_model)
    if (checkpoint / 'adapter/adapter_config.json').is_file():
        from peft import PeftModel
        model.action_head.model = PeftModel.from_pretrained(model.action_head.model, checkpoint / 'adapter')
        merge_lora(model)
        model.eval()
        model.save_pretrained(args.output_dir)
        AutoProcessor.from_pretrained(processor).save_pretrained(args.output_dir)
        write(args.output_dir / 'analysis_export.json', {'checkpoint': str(checkpoint),
              'settings': settings, 'processor_source': str(processor), 'format': 'peft-adapter'})
        print(f'Merged export: {args.output_dir}', flush=True)
        return
    targets = install_lora(model, settings['rank'], settings['alpha'])
    index = checkpoint / 'model.safetensors.index.json'
    paths = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    expected = model.state_dict()
    loaded = set()
    # Stream sharded full trainer weights; reject adapter-only or incompatible files.
    for name in paths:
        state = load_file(str(checkpoint / name), device='cpu')
        unexpected = set(state) - set(expected)
        if unexpected or loaded & set(state):
            raise ValueError(f'Unexpected/duplicate checkpoint keys: {sorted(unexpected)[:10]}')
        for key, value in state.items():
            if value.shape != expected[key].shape:
                raise ValueError(f'Shape mismatch: {key}')
            with torch.no_grad():
                expected[key].copy_(value)
        loaded.update(state)
        del state
    missing = set(expected) - loaded
    if missing:
        raise ValueError(f'Incomplete checkpoint; missing keys: {sorted(missing)[:10]}')
    if not any('.lora_B.default.weight' in k for k in loaded):
        raise ValueError('Checkpoint has no PEFT LoRA weights')
    merge_lora(model)
    model.eval()
    model.save_pretrained(args.output_dir)
    AutoProcessor.from_pretrained(processor).save_pretrained(args.output_dir)
    write(args.output_dir / 'analysis_export.json', {'checkpoint': str(checkpoint),
          'settings': settings, 'processor_source': str(processor), 'targets': targets})
    print(f'Merged export: {args.output_dir}', flush=True)


def trend(args):
    rows = []
    for path in sorted(args.results_root.rglob('evaluation.json')):
        settings = json.loads(path.read_text())
        match = re.search(r'checkpoint-(\d+)', settings['model_path'])
        for seed in settings['seeds']:
            log = path.parent / f'seed_{seed}.log'
            if not log.exists():
                raise ValueError(f'Missing log: {log}')
            text = log.read_text()
            values = re.findall(r'MSE for trajectory (\d+): ([\d.eE+-]+), MAE: ([\d.eE+-]+)', text)
            if sorted(int(v[0]) for v in values) != sorted(settings['traj_ids']) or 'INFO:root:Done' not in text:
                raise ValueError(f'Incomplete evaluation: {log}')
            rows.append({'run': path.parent.name, 'model': settings['model'],
                'step': int(match.group(1)) if match else '', 'seed': seed,
                'mse': sum(float(v[1]) for v in values)/len(values),
                'mae': sum(float(v[2]) for v in values)/len(values),
                'settings': {k: settings[k] for k in ('dataset_manifest_sha256', 'traj_ids', 'seeds',
                     'execution_horizon', 'steps', 'denoising_steps')}})
    if not rows:
        raise ValueError('No evaluation.json found')
    if any(r['settings'] != rows[0]['settings'] for r in rows):
        raise ValueError('Evaluation conditions differ; compare matched runs only')
    write(args.output_dir / 'trend.json', rows)
    with (args.output_dir / 'trend.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=['run','model','step','seed','mse','mae'])
        writer.writeheader()
        writer.writerows({k:v for k,v in row.items() if k != 'settings'} for row in rows)
    for path in args.training_root.glob('checkpoint-*/trainer_state.json'):
        state = json.loads(path.read_text())
        write(args.output_dir / (path.parent.name + '_log_history.json'), state.get('log_history', []))
    print(f'Trend rows: {len(rows)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groot-root', type=Path)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('equivalence', 'processor', 'export', 'trend'):
        p = sub.add_parser(name)
        p.add_argument('--output-dir', type=Path, required=True)
        if name in ('equivalence', 'processor', 'export'):
            p.add_argument('--base-model', type=Path, required=True)
        if name == 'processor': p.add_argument('--model-path', type=Path, required=True)
        if name == 'export': p.add_argument('--checkpoint', type=Path, required=True)
        if name == 'trend':
            p.add_argument('--results-root', type=Path, required=True)
            p.add_argument('--training-root', type=Path, required=True)
        if name == 'equivalence':
            p.add_argument('--dataset-root', type=Path, required=True)
            p.add_argument('--traj-ids', type=int, nargs='+', default=[0])
            p.add_argument('--frame-ids', type=int, nargs='+', default=[0, 16])
            p.add_argument('--rank', type=int, default=8)
            p.add_argument('--alpha', type=float, default=16)
            p.add_argument('--device', default='cuda:0')
            p.add_argument('--seed', type=int, default=42)
            p.add_argument('--denoising-steps', type=int, default=4)
            p.add_argument('--atol', type=float, default=1e-5)
            p.add_argument('--rtol', type=float, default=1e-5)
            p.add_argument('--tag', default='OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT')
    args = parser.parse_args()
    if args.mode == 'equivalence':
        if min(args.rank, args.denoising_steps) < 1 or args.alpha <= 0 or min(args.traj_ids + args.frame_ids) < 0 or min(args.atol, args.rtol) < 0:
            parser.error('Invalid LoRA, frame, trajectory or tolerance settings')
        if (args.output_dir / 'zero_lora_inference').exists():
            parser.error('zero_lora_inference already exists; use a new output directory')
    if args.groot_root: sys.path.insert(0, str(args.groot_root.resolve()))
    {'equivalence': equivalence, 'processor': compare_processors,
     'export': export_checkpoint, 'trend': trend}[args.mode](args)


if __name__ == '__main__':
    main()
