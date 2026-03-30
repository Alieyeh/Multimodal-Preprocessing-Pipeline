"""HAR helpers for harmonisation, resampling, cleaning, and metadata."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from mmprep.common.signal_utils import median_clip_fill, uniform_resample
from mmprep.common.windowing import _norm_label, sliding_windows

DEFAULT_HAR_LABEL_SCHEMA_NAME = "har_unified_v1"


def _source_token(source_name: str) -> str:
    """Convert a source record identifier into a sample-id-safe token."""
    token = re.sub(r"[^A-Za-z0-9]+", "_", source_name).strip("_").lower()
    return token or "source"


def clean_and_resample_har(
    df: pd.DataFrame,
    channel_schema: list[str],
    target_hz: int,
    source_hz: int | None = None,
    clip_zscore_at: float | None = None,
) -> pd.DataFrame:
    """Clean a harmonized HAR frame and resample it to the target rate.

    Inputs:
    - df: DataFrame with harmonized channel columns and a `label` column.
    - channel_schema: ordered signal channels to keep.
    - target_hz: required output sampling rate in Hz.
    - source_hz: source sampling rate in Hz when known. If omitted, data is assumed to
      already be at the target rate.
    - clip_zscore_at: optional symmetric clipping threshold after robust standardization.

    Outputs:
    - Cleaned and resampled DataFrame with the same schema plus label provenance fields.
    """
    out, qc = median_clip_fill(df, channel_schema, zmax=float(clip_zscore_at or 8.0))
    if source_hz and int(source_hz) != int(target_hz) and "time" not in out.columns:
        out["time"] = np.arange(len(out), dtype=np.float64) / float(source_hz)
    out = uniform_resample(
        out,
        channel_columns=channel_schema,
        target_hz=int(target_hz),
        time_column="time" if "time" in out.columns else None,
        label_column="label",
    )
    out["signal_qc_summary"] = f"missing_before={qc['missing_before']}|missing_after={qc['missing_after']}|n_clipped={qc['n_clipped']}"
    return out.reset_index(drop=True)


def harmonize_har_labels(
    df: pd.DataFrame,
    dataset_name: str,
    label_map: dict[str, str] | None = None,
    label_schema_name: str = DEFAULT_HAR_LABEL_SCHEMA_NAME,
) -> pd.DataFrame:
    """Map dataset-specific HAR labels into a unified schema with provenance.

    Inputs:
    - df: cleaned HAR DataFrame containing a `label` column.
    - dataset_name: source dataset name used for provenance and fallback lookup.
    - label_map: optional explicit mapping from original labels to harmonized labels.

    Outputs:
    - DataFrame with `label`, `original_label`, and `label_schema_name` columns.
    """
    out = df.copy()
    original = out["label"].astype(str).str.strip()
    normalized_map = {str(k).strip(): str(v).strip() for k, v in (label_map or {}).items()}
    null_like = {"0", "null", "transient", "none", "nan"}
    mapped = original.map(normalized_map)
    mapped = mapped.where(mapped.notna(), original.where(original.isin(null_like), "other"))
    out["original_label"] = original
    out["label"] = mapped
    out["label_schema_name"] = str(label_schema_name).strip() or DEFAULT_HAR_LABEL_SCHEMA_NAME
    out["source_dataset_name"] = dataset_name
    return out


def _window_original_label(
    df: pd.DataFrame,
    start: int,
    end: int,
    assigned_label: str | None,
    null_labels: set,
) -> str | None:
    """Return source-label provenance aligned with the window's assigned harmonized label."""
    if assigned_label is None or "original_label" not in df.columns:
        return None
    original = df["original_label"].iloc[start:end].astype(str)
    harmonized = df["label"].iloc[start:end].astype(str)
    mask = harmonized == str(assigned_label)
    candidates = original[mask]
    if candidates.empty:
        candidates = original[harmonized.map(_norm_label).map(lambda value: value not in {_norm_label(v) for v in null_labels})]
    if candidates.empty:
        return None
    return str(candidates.value_counts().sort_values(ascending=False).index[0])


def _standardize_har_windows(windows: np.ndarray) -> np.ndarray:
    """Apply per-window per-channel z-scoring to HAR windows."""
    if windows.size == 0:
        return windows.astype(np.float32, copy=False)
    mu = windows.mean(axis=2, keepdims=True)
    sigma = windows.std(axis=2, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    return ((windows - mu) / sigma).astype(np.float32, copy=False)


def make_har_windows(
    df: pd.DataFrame,
    dataset_name: str,
    subject_id: str,
    source_name: str,
    sampling_rate_hz: int,
    channel_schema: list[str],
    kind: str,
    null_labels: set,
    label_policy: str = "majority_non_null",
    standardize_per_window: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Convert one harmonised HAR record into final windows and metadata.

    Inputs:
    - df: DataFrame containing shared HAR channels and a `label` column.
    - dataset_name: source dataset name.
    - subject_id: subject identifier.
    - source_name: source file or record identifier.
    - sampling_rate_hz: shared target sampling rate.
    - channel_schema: ordered list of channel names.
    - kind: `pretrain` or `supervised`.
    - null_labels: set of labels treated as null/transient.

    Outputs:
    - Window tensor shaped [N, C, T].
    - One metadata dictionary per output window.
    """
    x = df[channel_schema].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy() if "label" in df.columns else None
    if kind == "pretrain":
        window_size = sampling_rate_hz * 10
        step_size = window_size
        windows, labels, starts = sliding_windows(x, None, window_size, step_size, return_starts=True)
    else:
        window_size = sampling_rate_hz * 5
        step_size = int(window_size / 2)
        windows, labels, starts = sliding_windows(
            x,
            y,
            window_size,
            step_size,
            null_labels=null_labels,
            label_policy=label_policy,
            return_starts=True,
        )
    if standardize_per_window:
        windows = _standardize_har_windows(windows)

    rows = []
    src_token = _source_token(source_name)
    for idx in range(windows.shape[0]):
        start = starts[idx]
        end = start + windows.shape[2]
        rows.append({
            "sample_id": f"{dataset_name}_{subject_id}_{src_token}_{kind}_{idx:05d}",
            "dataset_name": dataset_name,
            "modality": "har",
            "subject_or_patient_id": subject_id,
            "source_file_or_record": source_name,
            "split": "unspecified",
            "label_or_event": labels[idx],
            "sampling_rate_hz": sampling_rate_hz,
            "n_channels": len(channel_schema),
            "n_samples": windows.shape[2],
            "channel_schema": "|".join(channel_schema),
            "qc_flags": "",
            "window_kind": kind,
            "window_start_sample": start,
            "window_end_sample": end,
            "window_seconds": windows.shape[2] / float(sampling_rate_hz),
            "null_label_policy": label_policy,
            "original_label": None if kind == "pretrain" else _window_original_label(df, start, end, labels[idx], null_labels),
            "label_schema_name": df.get("label_schema_name", pd.Series([DEFAULT_HAR_LABEL_SCHEMA_NAME])).iloc[0],
            "signal_qc_summary": df.get("signal_qc_summary", pd.Series([""])).iloc[0],
        })
    return windows, rows
