"""Blend decoded GR00T N1.7/N1.6 DROID actions, without additional training.

Run both upstream servers with --use-sim-policy-wrapper. This process exposes
the N1.7 msgpack_numpy/ZeroMQ protocol. See README.md for commands and shapes.
Internal action_pred tensors are deliberately decoded by each upstream policy:
their normalization statistics, channel ordering and padding are different.
"""

import argparse
import copy
import io
import math
import time
from pathlib import Path

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq


ACTION_DIMS = {"joint_position": 7, "gripper_position": 1}


def encode(data, legacy=False):
    def default(value):
        if isinstance(value, np.ndarray) and value.dtype.hasobject:
            raise TypeError("Object arrays are not supported")
        if legacy and isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
        return mnp.encode(value)

    return msgpack.packb(data, default=default, use_bin_type=True)


def decode(payload):
    def hook(value):
        if value.get("__ndarray_class__"):
            return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
        if value.get("__ModalityConfig__") or value.get("__ModalityConfig_class__"):
            return value["as_json"]
        if value.get(b"nd", value.get("nd")) and value.get(b"kind", value.get("kind")) in (b"O", "O"):
            raise ValueError("Object arrays are not supported")
        return mnp.decode(value)

    return msgpack.unpackb(payload, object_hook=hook, raw=False)


class ModelClient:
    """Use each server's serializer; recreate REQ sockets after timeouts."""

    def __init__(self, endpoint, legacy=False, timeout_ms=120000):
        self.endpoint = endpoint
        self.legacy = legacy
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = None
        self._connect()

    def _connect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(self.endpoint)

    def call(self, endpoint, data=None):
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(encode(request, legacy=self.legacy))
            payload = self.socket.recv()
        except zmq.Again as exc:
            self._connect()
            raise TimeoutError(f"{self.endpoint}: {endpoint} timed out") from exc
        if payload == b"ERROR":
            raise RuntimeError(f"{self.endpoint}: server returned ERROR")
        response = decode(payload)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"{self.endpoint}: {response['error']}")
        return response

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


def blend_actions(actions17, actions16, alpha=0.5, horizon=8):
    """Mix absolute joint targets (radians) and continuous gripper targets.

    Arrays must refer to the same observation and future execution timestamps.
    EEF output and padded dimensions are not blended or returned.
    """
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    result = {}
    batch_size = None
    for name, dim in ACTION_DIMS.items():
        key = f"action.{name}"
        a17 = np.asarray(actions17[key])
        a16 = np.asarray(actions16[key])
        for label, arr in (("N1.7", a17), ("N1.6", a16)):
            if arr.ndim != 3 or arr.shape[2] != dim or arr.shape[0] == 0:
                raise ValueError(f"{label} {key}: expected [B,T,{dim}], got {arr.shape}")
            if arr.shape[1] < horizon:
                raise ValueError(f"{label} {key}: horizon {arr.shape[1]} < {horizon}")
            if arr.dtype.kind not in "fiu" or not np.isfinite(arr[:, :horizon]).all():
                raise ValueError(f"{label} {key}: expected finite numeric actions")
        if a17.shape[0] != a16.shape[0]:
            raise ValueError(f"{key}: model batch sizes differ")
        if batch_size is not None and a17.shape[0] != batch_size:
            raise ValueError("Joint and gripper batch sizes differ")
        batch_size = a17.shape[0]
        a17 = a17[:, :horizon].astype(np.float32)
        a16 = a16[:, :horizon].astype(np.float32)
        # Preserve the selected model exactly at the endpoints.
        result[key] = (a17.copy() if alpha == 1 else a16.copy() if alpha == 0
                       else alpha * a17 + (1 - alpha) * a16)
        if not np.isfinite(result[key]).all():
            raise ValueError(f"{key}: blending overflow")
    return result


def adapt_observation(observation, target, source):
    """Select actual timestamp offsets, never duplicate a missing video frame.

    Input is the flat N1.7 observation contract advertised by this proxy.
    """
    result = {}
    for group in ("video", "state"):
        source_offsets = source[group]["delta_indices"]
        target_offsets = target[group]["delta_indices"]
        try:
            indices = [source_offsets.index(offset) for offset in target_offsets]
        except ValueError as exc:
            raise ValueError(f"{group}: source offsets {source_offsets} cannot supply {target_offsets}") from exc
        for name in target[group]["modality_keys"]:
            key = f"{group}.{name}"
            arr = np.asarray(observation[key])
            rank = 5 if group == "video" else 3
            if arr.ndim != rank or arr.shape[1] != len(source_offsets):
                raise ValueError(f"{key}: expected rank {rank}, T={len(source_offsets)}, got {arr.shape}")
            result[key] = arr[:, indices].copy()
    source_language = source["language"]["modality_keys"][0]
    language = observation[source_language]
    if isinstance(language, np.ndarray):
        language = language.reshape(-1).tolist()
    if not isinstance(language, (tuple, list)) or not all(isinstance(item, str) for item in language):
        raise ValueError("Language must be a batch of strings")
    for name in target["language"]["modality_keys"]:
        result[name] = list(language)
    return result


class BlendedPolicy:
    def __init__(self, client17, client16, alpha=0.5, horizon=8):
        if not math.isfinite(alpha) or not 0 <= alpha <= 1 or horizon <= 0:
            raise ValueError("Require 0 <= alpha <= 1 and positive horizon")
        self.client17, self.client16 = client17, client16
        self.alpha, self.horizon = alpha, horizon
        self.config17 = client17.call("get_modality_config")
        self.config16 = client16.call("get_modality_config")
        for config in (self.config17, self.config16):
            action = config["action"]
            if not all(key in action["modality_keys"] for key in ACTION_DIMS):
                raise ValueError("Both servers must expose DROID joint_position and gripper_position")
            if len(action["delta_indices"]) < horizon:
                raise ValueError("Requested horizon exceeds upstream modality configuration")
        offsets17 = self.config17["action"]["delta_indices"][:horizon]
        offsets16 = self.config16["action"]["delta_indices"][:horizon]
        if offsets17 != offsets16:
            raise ValueError(f"Action time offsets differ: {offsets17} vs {offsets16}")
        # A consumer sends observations using N1.7's contract, including EEF state.
        # Check that this contract can also supply every N1.6 input.
        for group in ("video", "state"):
            for field in ("modality_keys", "delta_indices"):
                if not set(self.config16[group][field]) <= set(self.config17[group][field]):
                    raise ValueError(f"N1.7 {group}.{field} cannot supply N1.6 inputs")
        self.config = copy.deepcopy(self.config17)
        action = self.config["action"]
        indices = [action["modality_keys"].index(name) for name in ACTION_DIMS]
        if action.get("action_configs"):
            action["action_configs"] = [action["action_configs"][i] for i in indices]
        action["modality_keys"] = list(ACTION_DIMS)
        action["delta_indices"] = offsets17

    def get_action(self, observation, options=None):
        if options:
            raise ValueError("v1.0 supports ordinary action generation only; RTC options are unsupported")
        obs17 = adapt_observation(observation, self.config17, self.config17)
        obs16 = adapt_observation(observation, self.config16, self.config17)
        start = time.perf_counter()
        actions17, _ = self.client17.call("get_action", {"observation": obs17, "options": None})
        actions16, _ = self.client16.call("get_action", {"observation": obs16, "options": None})
        actions = blend_actions(actions17, actions16, self.alpha, self.horizon)
        return actions, {"alpha": self.alpha, "horizon": self.horizon,
                         "blend_space": "decoded_absolute_joint_and_gripper",
                         "latency_seconds": time.perf_counter() - start}

    def reset(self, options=None):
        return {"n17": self.client17.call("reset", {"options": options}),
                "n16": self.client16.call("reset", {"options": options})}

    def get_modality_config(self):
        return {name: {"__ModalityConfig__": True, "as_json": config}
                for name, config in self.config.items()}


def serve(policy, host, port):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    try:
        socket.bind(f"tcp://{host}:{port}")
        print(f"Protoss ready: tcp://{host}:{port}, alpha={policy.alpha}, horizon={policy.horizon}", flush=True)
        running = True
        while running:
            payload = socket.recv()
            try:
                request = decode(payload)
                endpoint = request.get("endpoint", "get_action")
                if endpoint == "ping":
                    response = {"status": "ok"}
                elif endpoint == "kill":
                    running, response = False, None
                elif endpoint == "get_modality_config":
                    response = policy.get_modality_config()
                elif endpoint in ("get_action", "reset"):
                    response = getattr(policy, endpoint)(**request.get("data", {}))
                else:
                    raise ValueError(f"Unknown endpoint: {endpoint}")
                reply = encode(response)
            except Exception as exc:
                print(f"Request failed: {exc}", flush=True)
                reply = encode({"error": str(exc)})
            socket.send(reply)
    finally:
        socket.close(linger=1000)
        context.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n17-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--n16-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--alpha", type=float, default=0.5, help="N1.7 weight; N1.6 weight is 1-alpha")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--observation", type=Path, help="Single flat observation .npz instead of server mode")
    parser.add_argument("--output", type=Path, default=Path("blended_actions.npz"))
    args = parser.parse_args()
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    client17 = ModelClient(args.n17_endpoint, timeout_ms=args.timeout_ms)
    client16 = ModelClient(args.n16_endpoint, legacy=True, timeout_ms=args.timeout_ms)
    try:
        policy = BlendedPolicy(client17, client16, args.alpha, args.horizon)
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
        client16.close()


if __name__ == "__main__":
    main()
