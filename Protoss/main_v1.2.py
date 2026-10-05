"""Refine blended GR00T N1.7 / pi05_droid_jointpos action chunks with residual flow.

alpha=1 selects N1.7 before refinement; alpha=0 selects pi0.5. Both servers are called.
The pi server must use pi05_droid_jointpos, which returns absolute joint
positions plus absolute gripper position (8 dimensions, in that order).
See ../README_v1.2.md for installation, model download and inference.
"""

import argparse
import copy
import importlib.util
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


# Share the existing GR00T transport/server without modifying the v1.0 script.
_spec = importlib.util.spec_from_file_location("protoss_v10_transport", Path(__file__).with_name("main_v1.0.py"))
_v10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v10)
ModelClient = _v10.ModelClient
serve = _v10.serve


class OpenPiClient:
    """OpenPI's codec over WebSocket with bounded connect/receive timeouts.

    A failed inference is surfaced rather than silently retried with a new
    stochastic action. The socket is closed and reconnected on the next call.
    """

    def __init__(self, uri="ws://127.0.0.1:8000", timeout_ms=120000):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.uri, self.timeout = uri, timeout_ms / 1000
        self.codec, self.connect = msgpack_numpy, connect
        self.socket = None
        self.metadata = {}
        self._connect()

    def _connect(self):
        try:
            self.socket = self.connect(
                self.uri, compression=None, max_size=None, open_timeout=self.timeout,
                # OpenPI can block its event loop during the first JAX compile.
                # Bound inference with recv(timeout=...) instead of heartbeat.
                ping_interval=None,
            )
            # OpenPI sends server metadata immediately after connecting.
            self.metadata = self._receive()
        except Exception:
            self.close()
            raise

    def _receive(self):
        response = self.socket.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server error: {response}")
        result = self.codec.unpackb(response)
        if not isinstance(result, dict):
            raise ValueError("OpenPI response must be a dictionary")
        if "error" in result:
            raise RuntimeError(f"OpenPI server error: {result['error']}")
        return result

    def infer(self, observation):
        try:
            if self.socket is None:
                self._connect()
            self.socket.send(self.codec.Packer().pack(observation))
            return self._receive()
        except Exception:
            self.close()
            raise

    def reset(self):
        # OpenPI's standard client reset is a no-op. There is no reset RPC.
        # Reconnect to discard any pending response; the model is stateless.
        self.close()
        return {"stateless": True}

    def close(self):
        if self.socket is not None:
            socket, self.socket = self.socket, None
            socket.close()


def pi_observations(observation, config):
    """Map each batch item to OpenPI's unbatched DROID input contract.

    Use the current (delta=0) camera/state sample. Match RoboLab's OpenPI
    letterboxing to 224x224; GR00T receives the original images separately.
    """
    from openpi_client.image_tools import resize_with_pad

    video_index = config["video"]["delta_indices"].index(0)
    state_index = config["state"]["delta_indices"].index(0)
    joint = np.asarray(observation["state.joint_position"])
    gripper = np.asarray(observation["state.gripper_position"])
    if joint.ndim != 3 or joint.shape[2] != 7 or joint.shape[0] == 0:
        raise ValueError("Joint state must be [B,T,7] with nonempty B")
    if gripper.ndim != 3 or gripper.shape[2] != 1 or gripper.shape[:2] != joint.shape[:2]:
        raise ValueError("Gripper state must be [B,T,1] matching joint state")
    if not np.isfinite(joint).all() or not np.isfinite(gripper).all():
        raise ValueError("State must be finite")
    batch = joint.shape[0]
    language = observation[config["language"]["modality_keys"][0]]
    if isinstance(language, np.ndarray):
        language = language.reshape(-1).tolist()
    if not isinstance(language, (list, tuple)) or len(language) != batch:
        raise ValueError("Language batch must match state batch")
    if not all(isinstance(prompt, str) for prompt in language):
        raise ValueError("Prompts must be strings")
    images = {}
    for name in ("exterior_image_1_left", "wrist_image_left"):
        arr = np.asarray(observation[f"video.{name}"])
        if arr.ndim != 5 or arr.shape[0] != batch or arr.shape[-1] != 3 or arr.dtype != np.uint8:
            raise ValueError(f"video.{name} must be RGB uint8 [B,T,H,W,3]")
        images[name] = resize_with_pad(arr[:, video_index], 224, 224)
    return [
        {
            "observation/exterior_image_1_left": images["exterior_image_1_left"][i],
            "observation/wrist_image_left": images["wrist_image_left"][i],
            "observation/joint_position": joint[i, state_index].astype(np.float32),
            "observation/gripper_position": gripper[i, state_index].astype(np.float32),
            "prompt": language[i],
        }
        for i in range(batch)
    ]


def pi_actions_to_dict(responses, horizon):
    """Split OpenPI's absolute [T,8] output into GR00T-style batched fields."""
    chunks = []
    for response in responses:
        chunk = np.asarray(response["actions"])
        if chunk.ndim != 2 or chunk.shape[1] != 8 or chunk.shape[0] < horizon:
            raise ValueError(f"pi05_droid_jointpos actions must be [T>=H,8], got {chunk.shape}")
        if chunk.dtype.kind not in "fiu" or not np.isfinite(chunk[:horizon]).all():
            raise ValueError("OpenPI actions must be finite numeric values")
        chunks.append(chunk[:horizon].astype(np.float32))
    if not chunks:
        raise ValueError("OpenPI response batch is empty")
    actions = np.stack(chunks)
    return {"action.joint_position": actions[..., :7],
            "action.gripper_position": actions[..., 7:8]}


def blend_actions(actions17, actions_pi, alpha=0.5, horizon=8):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or horizon <= 0:
        raise ValueError("Require 0 <= alpha <= 1 and positive horizon")
    result, batch = {}, None
    for name, dim in _v10.ACTION_DIMS.items():
        key = f"action.{name}"
        chunks = []
        for label, actions in (("N1.7", actions17), ("pi0.5", actions_pi)):
            arr = np.asarray(actions[key])
            if arr.ndim != 3 or arr.shape[2] != dim or arr.shape[1] < horizon or arr.shape[0] == 0:
                raise ValueError(f"{label} {key}: expected [B,T>=H,{dim}], got {arr.shape}")
            if batch is None:
                batch = arr.shape[0]
            if arr.shape[0] != batch:
                raise ValueError("Model/joint/gripper batch sizes differ")
            if arr.dtype.kind not in "fiu" or not np.isfinite(arr[:, :horizon]).all():
                raise ValueError(f"{label} {key}: expected finite numeric actions")
            chunks.append(arr[:, :horizon].astype(np.float32))
        a17, api = chunks
        if alpha == 1:
            result[key] = a17.copy()
        elif alpha == 0:
            result[key] = api.copy()
        else:
            result[key] = alpha * a17 + (1 - alpha) * api

        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key}: blending overflow")
    return result


class BlendedPolicy:
    def __init__(self, 
                 client_gr00t, 
                 client_pi,
                 alpha=0.5,
                 horizon=8, 
                 pi_horizon=15):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not 0 < horizon <= pi_horizon:
            raise ValueError("Require 0 <= alpha <= 1 and 0 < horizon <= pi_horizon")
        self.alpha        = alpha
        self.horizon      = horizon
        self.client_gr00t = client_gr00t
        self.client_pi    = client_pi
        self.config_gr00t = client_gr00t.call("get_modality_config")
        for group in ("video", "state"):
            if 0 not in self.config_gr00t[group]["delta_indices"]:
                raise ValueError(f"OpenPI needs current {group} sample (delta=0)")
        offsets = self.config_gr00t["action"]["delta_indices"][:horizon]
        if offsets != list(range(horizon)):
            raise ValueError("N1.7 must supply H consecutive actions beginning at delta=0")
        self.config = copy.deepcopy(self.config_gr00t)
        action = self.config["action"]
        indices = [action["modality_keys"].index(name) for name in _v10.ACTION_DIMS]
        if action.get("action_configs"):
            action["action_configs"] = [action["action_configs"][i] for i in indices]
        action["modality_keys"] = list(_v10.ACTION_DIMS)
        action["delta_indices"] = offsets

    def get_action(self, observation, options=None):
        if options:
            raise ValueError("v1.2 supports ordinary action generation only; RTC is unsupported")
        obs_gr00t   = _v10.adapt_observation(observation, self.config_gr00t, self.config_gr00t)
        requests_pi = pi_observations(obs_gr00t, self.config_gr00t)
        start       = time.perf_counter()

        actions_gr00t, _ = self.client_gr00t.call("get_action", {"observation": obs_gr00t, "options": None})
        actions_pi       = pi_actions_to_dict([self.client_pi.infer(request) for request in requests_pi], self.horizon)
        actions_blended  = blend_actions(actions_gr00t, actions_pi, self.alpha, self.horizon)
        if getattr(self, "refiner", None) is not None:
            actions_output = self.refiner.refine(actions_blended, current_state(obs_gr00t, self.config_gr00t))
        else:
            actions_output = actions_blended
        return actions_output, {
            "alpha": self.alpha, "horizon": self.horizon,
            "models": ["GR00T-N1.7-DROID", "pi05_droid_jointpos"],
            "blend_space": "decoded_absolute_joint_and_gripper",
            "learnable_network": "residual_flow" if getattr(self, "refiner", None) else None,
            "latency_seconds": time.perf_counter() - start,
        }

    def reset(self, options=None):
        return {"n17": self.client_gr00t.call("reset", {"options": options}),
                "pi05": self.client_pi.reset()}

    def get_modality_config(self):
        return {name: {"__ModalityConfig__": True, "as_json": config}
                for name, config in self.config.items()}


def pack_actions(actions):
    return np.concatenate([actions["action.joint_position"],
                           actions["action.gripper_position"]], axis=-1)


def current_state(observation, config):
    i = config["state"]["delta_indices"].index(0)
    return np.concatenate([observation["state.joint_position"][:, i],
                           observation["state.gripper_position"][:, i]], axis=-1)


class ResidualFlow(nn.Module):
    """Conditional chunk velocity field; only this MLP is trainable."""
    def __init__(self, horizon=8, hidden=256):
        super().__init__()
        if horizon <= 0 or hidden <= 0:
            raise ValueError("horizon and hidden must be positive")
        self.horizon, self.hidden = horizon, hidden
        self.net = nn.Sequential(nn.Linear(16 * horizon + 9, hidden), 
                                 nn.SiLU(),
                                 nn.Linear(hidden, hidden), 
                                 nn.SiLU(),
                                 nn.Linear(hidden, 8 * horizon))
        self.register_buffer("mean", torch.zeros(8))
        self.register_buffer("scale", torch.ones(8))

    def forward(self, noisy, blend, state, time):
        inputs = torch.cat([noisy.flatten(1), blend.flatten(1), state, time[:, None]], dim=-1)
        return self.net(inputs).reshape(-1, self.horizon, 8)

    def loss(self, blend, state, target, mask):
        # OpenPI convention: t=1 noise, t=0 data; residual in normalized units.
        residual = (target - blend) / self.scale
        noise = torch.randn_like(residual)
        t = torch.distributions.Beta(1.5, 1.0).sample((len(blend),)).to(blend.device)
        t = t * 0.999 + 0.001
        noisy = t[:, None, None] * noise + (1 - t[:, None, None]) * residual
        # Hide padded targets from all chunk features, not only from the loss.
        noisy = noisy * mask[..., None]
        velocity = self(noisy, (blend - self.mean) / self.scale,
                        (state - self.mean) / self.scale, t)
        error = (velocity - (noise - residual)).square() * mask[..., None]
        return error.sum() / (mask.sum() * 8).clamp_min(1)

    @torch.inference_mode()
    def refine(self, actions, state):
        device = self.mean.device
        blend = torch.as_tensor(pack_actions(actions), dtype=torch.float32, device=device)
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if blend.shape[1:] != (self.horizon, 8) or state.shape != (len(blend), 8):
            raise ValueError("Checkpoint horizon/state shape mismatch")
        x = torch.randn_like(blend)
        for i in range(self.steps):
            t = torch.full((len(blend),), 1 - i / self.steps, device=device)
            x = x - self(x, (blend - self.mean) / self.scale,
                         (state - self.mean) / self.scale, t) / self.steps
        output = (blend + x * self.scale).cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite refined actions")
        return {"action.joint_position": output[..., :7],
                "action.gripper_position": output[..., 7:]}


def load_refiner(path, alpha, horizon, device, steps):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format"] != "protoss_residual_flow_v1.2":
        raise ValueError("Unsupported checkpoint")
    if checkpoint["alpha"] != alpha or checkpoint["horizon"] != horizon:
        raise ValueError("Checkpoint alpha/horizon must match inference")
    model = ResidualFlow(horizon, checkpoint["hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    if not torch.isfinite(model.mean).all() or not torch.isfinite(model.scale).all() or (model.scale <= 0).any():
        raise ValueError("Invalid checkpoint normalization")
    model.steps = steps
    return model.eval()


def cache_samples(policy, input_dir, output_dir, resume=False):
    """Each source NPZ is one observation plus a demonstrated target chunk."""
    paths = sorted(input_dir.rglob("*.npz"))
    if not paths:
        raise ValueError("No source NPZ samples")
    if input_dir.resolve() == output_dir.resolve() or input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Cache directory must be outside input directory")
    if getattr(policy, "refiner", None) is not None:
        raise ValueError("Training cache requires unrefined pretrained blending")
    for index, path in enumerate(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        dest = output_dir / path.relative_to(input_dir)
        if dest.exists():
            if not resume:
                raise ValueError(f"Cache exists: {dest}; use --resume-cache or a new directory")
            with np.load(dest, allow_pickle=False) as existing:
                if (str(existing["source_sha256"]) != digest or
                    float(existing["alpha"]) != policy.alpha or int(existing["horizon"]) != policy.horizon):
                    raise ValueError(f"Existing cache does not match input/settings: {dest}")
            continue
        with np.load(path, allow_pickle=False) as data:
            observation = {k: data[k] for k in data.files if not k.startswith("target.") and k not in ("mask", "episode_id")}
            target = np.concatenate([data["target.joint_position"], data["target.gripper_position"]], axis=-1)
            mask = data["mask"]
            episode = data["episode_id"]
        blend, _ = policy.get_action(observation)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary = dest.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, blend=pack_actions(blend),
                            state=current_state(observation, policy.config_gr00t),
                            target=target, mask=mask, episode_id=episode,
                            alpha=policy.alpha, horizon=policy.horizon, source_sha256=digest)
        temporary.replace(dest)
        print(f"cache {index + 1}/{len(paths)}: {dest}", flush=True)
    print(f"Cached {len(paths)} files in {output_dir}")


def read_cache(directory, alpha, horizon):
    arrays = {k: [] for k in ("blend", "state", "target", "mask")}
    episodes = set()
    for path in sorted(directory.rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if float(data["alpha"]) != alpha or int(data["horizon"]) != horizon:
                raise ValueError(f"Cache alpha/horizon mismatch: {path}")
            values = {k: np.asarray(data[k], dtype=np.float32) for k in arrays}
            b = len(values["blend"])
            expected = {"blend": (b, horizon, 8), "target": (b, horizon, 8),
                        "state": (b, 8), "mask": (b, horizon)}
            if b == 0 or any(v.shape != expected[k] or not np.isfinite(v).all() for k, v in values.items()):
                raise ValueError(f"Invalid cache shapes/values: {path}")
            mask = values["mask"]
            if not np.isin(mask, [0, 1]).all() or (mask.sum(1) == 0).any():
                raise ValueError("Each sample needs a nonempty binary padding mask")
            ids = np.asarray(data["episode_id"])
            if ids.shape != (b,) or ids.dtype.kind not in "USiu":
                raise ValueError("episode_id must contain B string/integer episode IDs")
            episodes.update(str(x) for x in ids)
            for k, v in values.items():
                arrays[k].append(v)
    if not arrays["blend"]:
        raise ValueError(f"Empty cache: {directory}")
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in arrays.items()}, episodes


def train_refiner(args):
    if not args.train_cache or not args.val_cache:
        raise ValueError("Training needs --train-cache and --val-cache")
    torch.manual_seed(args.seed)
    train, train_ids = read_cache(args.train_cache, args.alpha, args.horizon)
    val, val_ids = read_cache(args.val_cache, args.alpha, args.horizon)
    if train_ids & val_ids:
        raise ValueError("Train/validation episodes overlap")
    model = ResidualFlow(args.horizon, args.hidden).to(args.device)
    valid_targets = train["target"][train["mask"].bool()]
    model.mean.copy_(valid_targets.mean(0).to(args.device))
    model.scale.copy_(valid_targets.std(0, unbiased=False).clamp_min(1e-3).to(args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(train[k] for k in ("blend", "state", "target", "mask"))),
        batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*(val[k] for k in ("blend", "state", "target", "mask"))),
        batch_size=args.batch_size)
    best = float("inf")
    history = []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_score, train_count = 0., 0.
        for batch in loader:
            optimizer.zero_grad()
            loss = model.loss(*(x.to(args.device) for x in batch))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weight = float(batch[-1].sum())
            train_score += float(loss.detach()) * weight
            train_count += weight
        model.eval()
        # Fixed validation noise/time across epochs, without disturbing training RNG.
        with torch.random.fork_rng(devices=[torch.device(args.device).index or 0] if torch.device(args.device).type == "cuda" else []):
            torch.manual_seed(args.seed + 1)
            with torch.no_grad():
                score, count = 0.0, 0.0
                for batch in val_loader:
                    weight = float(batch[-1].sum())
                    score += float(model.loss(*(x.to(args.device) for x in batch))) * weight
                    count += weight
                score /= count
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation loss")
        print(f"epoch={epoch + 1} val_flow_mse={score:.6f}")
        history.append({"epoch": epoch + 1, "train_flow_mse": train_score / train_count,
                        "val_flow_mse": score})
        args.checkpoint.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
        if score < best:
            best = score
            torch.save({"format": "protoss_residual_flow_v1.2", "model": model.state_dict(),
                        "alpha": args.alpha, "horizon": args.horizon, "hidden": args.hidden,
                        "val_flow_mse": best, "seed": args.seed,
                        "train_episode_ids": sorted(train_ids), "val_episode_ids": sorted(val_ids)}, args.checkpoint)
    print(f"Saved best checkpoint: {args.checkpoint}")


def action_metrics(prediction, target, mask, scale):
    """Mask padding; report joint radians separately from continuous gripper."""
    error = prediction - target
    valid = mask.bool()
    joint_error = error[..., :7][valid]
    gripper_error = error[..., 7][valid]
    normalized_error = (error / scale)[valid]
    return {"joint_mae_rad": float(joint_error.abs().mean()),
            "joint_rmse_rad": float(joint_error.square().mean().sqrt()),
            "gripper_mae": float(gripper_error.abs().mean()),
            "gripper_rmse": float(gripper_error.square().mean().sqrt()),
            "normalized_action_mse": float(normalized_error.square().mean())}


def evaluate_refiner(args):
    if not args.test_cache:
        raise ValueError("Evaluation needs --test-cache (held-out episodes)")
    data, test_ids = read_cache(args.test_cache, args.alpha, args.horizon)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "train_episode_ids" not in checkpoint or "val_episode_ids" not in checkpoint:
        raise ValueError("Checkpoint lacks split provenance; retrain with the current v1.2 trainer")
    seen = set(checkpoint["train_episode_ids"]) | set(checkpoint["val_episode_ids"])
    if test_ids & seen:
        raise ValueError("Test episodes overlap training/validation episodes")
    model = load_refiner(args.checkpoint, args.alpha, args.horizon, args.device, args.flow_steps)
    baseline = action_metrics(data["blend"], data["target"], data["mask"], model.scale.cpu())
    runs = []
    for seed in args.eval_seeds:
        torch.manual_seed(seed)
        predictions = []
        started = time.perf_counter()
        for start in range(0, len(data["blend"]), args.batch_size):
            blend = data["blend"][start:start + args.batch_size].numpy()
            state = data["state"][start:start + args.batch_size].numpy()
            actions = {"action.joint_position": blend[..., :7],
                       "action.gripper_position": blend[..., 7:]}
            predictions.append(torch.from_numpy(pack_actions(model.refine(actions, state))))
        metrics = action_metrics(torch.cat(predictions), data["target"], data["mask"], model.scale.cpu())
        metrics["refinement_seconds_per_chunk"] = (time.perf_counter() - started) / len(data["blend"])
        runs.append({"seed": seed, **metrics})
    summary = {key: {"mean": float(np.mean([run[key] for run in runs])),
                     "std": float(np.std([run[key] for run in runs]))}
               for key in runs[0] if key != "seed"}
    report = {"evaluation": "held_out_demonstration_action_error",
              "checkpoint": str(args.checkpoint), "test_cache": str(args.test_cache),
              "alpha": args.alpha, "horizon": args.horizon, "flow_steps": args.flow_steps,
              "device": args.device, "batch_size": args.batch_size,
              "episodes": len(test_ids), "chunks": len(data["blend"]),
              "valid_timesteps": int(data["mask"].sum()), "test_episode_ids": sorted(test_ids),
              "v1.1_blend": baseline, "v1.2_refined": summary, "per_seed": runs,
              "note": "Action error on recorded observations; not closed-loop task success."}
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Saved evaluation: {args.metrics_output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n17-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--pi-uri", default="ws://127.0.0.1:8000")
    parser.add_argument("--alpha", type=float, default=0.5, help="N1.7 weight; pi0.5 weight is 1-alpha")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--pi-horizon", type=int, default=15, help="Serving config horizon; outputs are also validated")
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--observation", type=Path, help="Single flat observation NPZ instead of proxy server")
    parser.add_argument("--output", type=Path, default=Path("refined_actions_v1.2.npz"))
    parser.add_argument("--mode", choices=["infer", "cache", "train", "eval"], default="infer")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/refiner_v1.2.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--flow-steps", type=int, default=10)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--resume-cache", action="store_true", help="Reuse caches with matching source hashes/settings")
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--val-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--metrics-output", type=Path, default=Path("results/v1.2/offline_test.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.timeout_ms <= 0 or not 0 < args.horizon <= args.pi_horizon:
        parser.error("Require positive timeout and 0 < horizon <= pi-horizon")
    if min(args.flow_steps, args.epochs, args.batch_size, args.hidden) <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Training/flow sizes and learning rate must be positive")
    if args.mode == "train":
        train_refiner(args)
        return
    if args.mode == "eval":
        evaluate_refiner(args)
        return
    if args.mode == "cache" and (not args.input_dir or not args.cache_dir):
        parser.error("cache needs --input-dir and --cache-dir")
    refiner = None
    if args.mode == "infer":
        torch.manual_seed(args.seed)
        refiner = load_refiner(args.checkpoint, args.alpha, args.horizon, args.device, args.flow_steps)
    client_gr00t = ModelClient(args.n17_endpoint, timeout_ms=args.timeout_ms)
    client_pi = None
    try:
        client_pi = OpenPiClient(args.pi_uri, args.timeout_ms)
        policy = BlendedPolicy(client_gr00t, client_pi, args.alpha, args.horizon, args.pi_horizon)
        policy.refiner = refiner
        if args.mode == "cache":
            cache_samples(policy, args.input_dir, args.cache_dir, args.resume_cache)
        elif args.observation:
            with np.load(args.observation, allow_pickle=False) as archive:
                observation = {key: archive[key] for key in archive.files}
            actions, info = policy.get_action(observation)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.output, **actions)
            print(f"Saved {args.output}: {info}")
        else:
            serve(policy, args.host, args.port)
    finally:
        client_gr00t.close()
        if client_pi is not None:
            client_pi.close()


if __name__ == "__main__":
    main()
