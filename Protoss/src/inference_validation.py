"""Check inference values before numerical rotation conversion obscures the cause."""

import numpy as np


def require_finite(value, name):
    array = np.asarray(value)
    invalid = np.argwhere(~np.isfinite(array))
    if not np.isfinite(array).all():
        index = tuple(int(i) for i in invalid[0])
        raise ValueError(f"{name} contains NaN/Inf at index {index}; shape={array.shape}. "
                         "Check simulator observations, model weights and inference numerics.")
    return array


def decode_action_checked(processor, normalized_action, embodiment_tag, states):
    for name, value in states.items():
        require_finite(value, f"Reference state '{name}'")
    require_finite(normalized_action, "Model action_pred (before unnormalization)")
    return processor.decode_action(normalized_action, embodiment_tag, states)


def rotation6d_to_matrix(value):
    rows = require_finite(np.asarray(value, dtype=np.float64), "EEF rot6d").reshape(2, 3)
    # Rescale before computing norms to avoid overflow/underflow on finite values.
    scales = np.max(np.abs(rows), axis=1)
    if np.any(scales == 0):
        raise ValueError(f"Degenerate EEF rot6d: zero rotation row; values={rows.ravel().tolist()}")
    first, second = rows / scales[:, None]
    first /= np.linalg.norm(first)
    second -= np.dot(first, second) * first
    norm = np.linalg.norm(second)
    if norm <= 1e-12:
        raise ValueError(f"Degenerate EEF rot6d: parallel rotation rows; values={rows.ravel().tolist()}")
    second /= norm
    return np.stack((first, second, np.cross(first, second)))
