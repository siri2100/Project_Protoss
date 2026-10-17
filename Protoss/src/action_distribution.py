"""Plot denormalized inference actions, separately for each action dimension."""
import argparse
import copy
import gc
from pathlib import Path
import sys


def plot_distributions(before, after, output_dir, bins=40):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not before or before.keys() != after.keys():
        raise ValueError("Action keys are empty or differ between models")
    series = []
    for key in sorted(before):
        left, right = np.asarray(before[key]), np.asarray(after[key])
        if left.shape != right.shape or left.ndim < 2 or not left.size:
            raise ValueError(f"Action shape mismatch or empty predictions: {key}")
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Non-finite action predictions: {key}")
        for dim in range(left.shape[-1]):
            series.append((f"{key}[{dim}]", left[..., dim].ravel(), right[..., dim].ravel()))
    columns = min(3, len(series))
    rows = (len(series) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.5 * rows), squeeze=False)
    try:
        for axis, (label, left, right) in zip(axes.flat, series):
            edges = np.histogram_bin_edges(np.concatenate([left, right]), bins=bins)
            axis.hist(left, bins=edges, density=True, histtype="step", label="Pretrained", linewidth=1.6)
            axis.hist(right, bins=edges, density=True, histtype="step", label="Trained LoRA", linewidth=1.6)
            axis.set(title=label, xlabel="Action value (denormalized)", ylabel="Density")
            axis.legend()
        for axis in list(axes.flat)[len(series):]:
            axis.set_visible(False)
        fig.suptitle("Pretrained vs trained: matched observations and random seeds")
        fig.tight_layout()
        output = Path(output_dir) / "distribution_plot.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
    finally:
        plt.close(fig)
    print(f"Distribution plot: {output}", flush=True)
    return output


def compare(args):
    import numpy as np
    import torch
    from transformers import set_seed
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from .dataset_paths import split_path
    from .inference_loading import resolve_inference_paths

    _, adapter, _ = resolve_inference_paths(args.model_path, args.base_model)
    if adapter is None:
        raise ValueError("--model-path must identify a saved PEFT adapter run")
    policy = Gr00tPolicy(embodiment_tag=args.tag, model_path=str(args.base_model), device=args.device)
    configs = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(split_path(args.dataset_root, args.split)),
                                 modality_configs=configs)
    obs_configs = copy.deepcopy(configs)
    obs_configs.pop("action")
    observations = []
    for traj_id in args.traj_ids:
        traj = loader[traj_id]
        if not len(traj):
            raise ValueError(f"Empty trajectory: {traj_id}")
        frames = np.unique(np.linspace(0, len(traj) - 1, min(args.frames_per_trajectory, len(traj)), dtype=int))
        for step in frames:
            point = extract_step_data(traj, int(step), obs_configs, policy.embodiment_tag)
            obs = {f"state.{key}": value for key, value in point.states.items()}
            obs.update({f"video.{key}": np.array(value) for key, value in point.images.items()})
            obs.update({key: point.text for key in configs["language"].modality_keys})
            observations.append(parse_observation_gr00t(obs, configs))

    def predict(current):
        current.model.eval()
        current.model.action_head.num_inference_timesteps = args.denoising_steps
        predictions = {}
        for index, observation in enumerate(observations):
            set_seed(args.seed + index)
            with torch.inference_mode():
                actions, _ = current.get_action(copy.deepcopy(observation))
            if index and predictions.keys() != actions.keys():
                raise ValueError("Action keys changed between samples")
            for key, value in actions.items():
                predictions.setdefault(key, []).append(np.asarray(value).copy())
        return {key: np.stack(values) for key, values in predictions.items()}

    print(f"Comparing {len(observations)} observations from split={args.split}", flush=True)
    before = predict(policy)
    del policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    policy = Gr00tPolicy(embodiment_tag=args.tag, model_path=str(args.model_path),
                         base_model_path=str(args.base_model), lora_mode=args.lora_mode, device=args.device)
    if configs != policy.get_modality_config():
        raise ValueError("Modality configurations differ; identical observations cannot be compared")
    after = predict(policy)
    return plot_distributions(before, after, args.output_dir, args.bins)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--traj-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--frames-per-trajectory", type=int, default=32)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--lora-mode", choices=("unmerged", "merged"), default="unmerged")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
    args = parser.parse_args()
    if min(args.frames_per_trajectory, args.bins, args.denoising_steps) < 1 or min(args.traj_ids) < 0:
        parser.error("Sample counts, bins and denoising steps must be positive; trajectory IDs must be nonnegative")
    from .gr00t_adapter_patch import ensure_adapter_inference
    ensure_adapter_inference(args.groot_root)
    sys.path.insert(0, str(args.groot_root.resolve()))
    compare(args)
