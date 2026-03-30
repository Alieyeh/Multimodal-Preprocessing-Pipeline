"""Reusable window generation utilities."""
from __future__ import annotations

from collections import Counter

import numpy as np


def _norm_label(value: object) -> str:
    """Normalize labels for null-label comparison."""
    return str(value).strip().lower()


def sliding_windows(
    signal: np.ndarray,
    labels: np.ndarray | None,
    window_size: int,
    step_size: int,
    null_labels: set | None = None,
    label_policy: str = "majority_non_null",
    return_starts: bool = False,
) -> tuple[np.ndarray, list[str | None]] | tuple[np.ndarray, list[str | None], list[int]]:
    """Create sliding windows over a signal tensor.

    Inputs:
    - signal: array with shape [T, C].
    - labels: optional label vector of length T.
    - window_size: window length in timesteps.
    - step_size: step size in timesteps.
    - null_labels: labels treated as null for supervised majority voting.
    - label_policy: current supported value is `majority_non_null`.

    Outputs:
    - Window tensor shaped [N, C, T_window].
    - List of per-window labels or None values.
    - Optional list of retained window start indices.
    """
    if label_policy != "majority_non_null":
        raise ValueError(f"Unsupported HAR label_policy: {label_policy}")
    null_labels = {_norm_label(v) for v in (null_labels or set())}
    windows = []
    out_labels = []
    starts = []
    for start in range(0, max(signal.shape[0] - window_size + 1, 0), step_size):
        end = start + window_size
        segment = signal[start:end]
        if labels is None:
            windows.append(segment.T)
            out_labels.append(None)
            starts.append(start)
        else:
            valid = [x for x in labels[start:end].tolist() if _norm_label(x) not in null_labels]
            if not valid:
                continue
            else:
                windows.append(segment.T)
                out_labels.append(Counter(valid).most_common(1)[0][0])
                starts.append(start)
    if not windows:
        empty = np.empty((0, signal.shape[1], window_size), dtype=np.float32)
        if return_starts:
            return empty, [], []
        return empty, []
    stacked = np.stack(windows).astype(np.float32)
    if return_starts:
        return stacked, out_labels, starts
    return stacked, out_labels
