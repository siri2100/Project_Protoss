"""Blend decoded GR00T N1.7 with OpenPI pi05_droid_jointpos actions.

alpha=1 selects N1.7; alpha=0 selects pi0.5. Both servers are called.
The pi server must use pi05_droid_jointpos, which returns absolute joint
positions plus absolute gripper position (8 dimensions, in that order).
See ../README_v1.1.md for installation, model download and inference.
"""

import argparse
import copy
import importlib.util
import math
from pathlib import Path
import time

import numpy as np


# Share the existing GR00T transport/server without modifying the v1.0 script.
_spec = importlib.util.spec_from_file_location(
    "protoss_v10_transport", Path(__file__).with_name("main_v1.0.py")
)
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
        result[key] = (a17.copy() if alpha == 1 else api.copy() if alpha == 0
                       else alpha * a17 + (1 - alpha) * api)
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key}: blending overflow")
    return result


class BlendedPolicy:
    def __init__(self, client17, client_pi, alpha=0.5, horizon=8, pi_horizon=15):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1 or not 0 < horizon <= pi_horizon:
            raise ValueError("Require 0 <= alpha <= 1 and 0 < horizon <= pi_horizon")
        self.client17, self.client_pi = client17, client_pi
        self.alpha, self.horizon = alpha, horizon
        self.config17 = client17.call("get_modality_config")
        for group in ("video", "state"):
            if 0 not in self.config17[group]["delta_indices"]:
                raise ValueError(f"OpenPI needs current {group} sample (delta=0)")
        offsets = self.config17["action"]["delta_indices"][:horizon]
        if offsets != list(range(horizon)):
            raise ValueError("N1.7 must supply H consecutive actions beginning at delta=0")
        self.config = copy.deepcopy(self.config17)
        action = self.config["action"]
        indices = [action["modality_keys"].index(name) for name in _v10.ACTION_DIMS]
        if action.get("action_configs"):
            action["action_configs"] = [action["action_configs"][i] for i in indices]
        action["modality_keys"] = list(_v10.ACTION_DIMS)
        action["delta_indices"] = offsets

    def get_action(self, observation, options=None):
        if options:
            raise ValueError("v1.1 supports ordinary action generation only; RTC is unsupported")
        obs17 = _v10.adapt_observation(observation, self.config17, self.config17)
        requests_pi = pi_observations(obs17, self.config17)
        start = time.perf_counter()
        actions17, _ = self.client17.call("get_action", {"observation": obs17, "options": None})
        # OpenPI serves unbatched requests. Never add a batch axis to its input.
        responses = [self.client_pi.infer(request) for request in requests_pi]
        actions_pi = pi_actions_to_dict(responses, self.horizon)
        actions = blend_actions(actions17, actions_pi, self.alpha, self.horizon)
        return actions, {
            "alpha": self.alpha, "horizon": self.horizon,
            "models": ["GR00T-N1.7-DROID", "pi05_droid_jointpos"],
            "blend_space": "decoded_absolute_joint_and_gripper",
            "latency_seconds": time.perf_counter() - start,
        }

    def reset(self, options=None):
        return {"n17": self.client17.call("reset", {"options": options}),
                "pi05": self.client_pi.reset()}

    def get_modality_config(self):
        return {name: {"__ModalityConfig__": True, "as_json": config}
                for name, config in self.config.items()}


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
    parser.add_argument("--output", type=Path, default=Path("blended_actions_v1.1.npz"))
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    if args.timeout_ms <= 0 or not 0 < args.horizon <= args.pi_horizon:
        parser.error("Require positive timeout and 0 < horizon <= pi-horizon")
    client17 = ModelClient(args.n17_endpoint, timeout_ms=args.timeout_ms)
    client_pi = None
    try:
        client_pi = OpenPiClient(args.pi_uri, args.timeout_ms)
        policy = BlendedPolicy(client17, client_pi, args.alpha, args.horizon, args.pi_horizon)
        if args.observation:
            with np.load(args.observation, allow_pickle=False) as archive:
                observation = {key: archive[key] for key in archive.files}
            actions, info = policy.get_action(observation)
            np.savez(args.output, **actions)
            print(f"Saved {args.output}: {info}")
        else:
            serve(policy, args.host, args.port)
    finally:
        client17.close()
        if client_pi is not None:
            client_pi.close()


if __name__ == "__main__":
    main()
