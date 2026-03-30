"""Lightweight signal-processing helpers shared across modalities."""
from __future__ import annotations

from math import gcd

import numpy as np
import pandas as pd


def coerce_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values with NaN on parse failure.

    Inputs:
    - df: input table.
    - columns: sensor channel columns expected to be numeric.

    Outputs:
    - DataFrame copy with numeric channel columns.
    """
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def median_clip_fill(df: pd.DataFrame, columns: list[str], zmax: float = 8.0) -> tuple[pd.DataFrame, dict]:
    """Apply light robust cleaning to sensor channels.

    Inputs:
    - df: input sensor table.
    - columns: ordered signal columns.
    - zmax: robust z-score clipping threshold.

    Outputs:
    - Cleaned DataFrame.
    - QC summary dictionary with clipping and missing-value counts.
    """
    out = coerce_numeric_frame(df, columns)
    missing_before = int(out[columns].isna().sum().sum())
    n_clipped = 0
    for col in columns:
        series = out[col].astype(float)
        med = float(np.nanmedian(series)) if np.isfinite(series).any() else 0.0
        mad = float(np.nanmedian(np.abs(series - med))) if np.isfinite(series).any() else 0.0
        scale = 1.4826 * mad if mad > 1e-9 else float(np.nanstd(series))
        if not np.isfinite(scale) or scale < 1e-9:
            scale = 1.0
        lo = med - zmax * scale
        hi = med + zmax * scale
        n_clipped += int(((series < lo) | (series > hi)).fillna(False).sum())
        out[col] = series.clip(lo, hi)
    out[columns] = out[columns].interpolate(limit_direction="both").ffill().bfill()
    missing_after = int(out[columns].isna().sum().sum())
    return out, {
        "missing_before": missing_before,
        "missing_after": missing_after,
        "n_clipped": n_clipped,
    }


def uniform_resample(
    df: pd.DataFrame,
    channel_columns: list[str],
    target_hz: int,
    time_column: str | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Resample a table onto a uniform target grid.

    Inputs:
    - df: source table.
    - channel_columns: signal columns to interpolate.
    - target_hz: destination sampling rate in Hz.
    - time_column: optional timestamp column.
    - label_column: optional label column copied by nearest-neighbor lookup.

    Outputs:
    - Resampled DataFrame with channels and optional labels.
    """
    if df.empty:
        cols = list(channel_columns) + ([label_column] if label_column else [])
        return pd.DataFrame(columns=[c for c in cols if c is not None])

    if time_column and time_column in df.columns:
        times = pd.to_numeric(df[time_column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times)
        base = df.loc[valid].reset_index(drop=True).copy()
        times = times[valid]
        if len(times) < 2:
            cols = channel_columns + ([label_column] if label_column and label_column in base.columns else [])
            return base[cols].reset_index(drop=True)
        order = np.argsort(times)
        base = base.iloc[order].reset_index(drop=True)
        times = times[order]
        start = float(times[0])
        end = float(times[-1])
        if end <= start:
            cols = channel_columns + ([label_column] if label_column and label_column in base.columns else [])
            return base[cols].reset_index(drop=True)
        grid = np.arange(start, end + 1e-12, 1.0 / float(target_hz))
        payload = {
            col: np.interp(grid, times, base[col].to_numpy(dtype=float))
            for col in channel_columns
        }
        out = pd.DataFrame(payload)
        if label_column and label_column in base.columns:
            idx = np.clip(np.searchsorted(times, grid, side="left"), 0, len(times) - 1)
            prev_idx = np.clip(idx - 1, 0, len(times) - 1)
            choose_prev = np.abs(grid - times[prev_idx]) <= np.abs(grid - times[idx])
            nearest = np.where(choose_prev, prev_idx, idx)
            out[label_column] = base[label_column].iloc[nearest].to_numpy()
        return out.reset_index(drop=True)

    cols = channel_columns + ([label_column] if label_column and label_column in df.columns else [])
    return df[cols].reset_index(drop=True).copy()


def resample_array(signal: np.ndarray, source_hz: float, target_hz: int) -> np.ndarray:
    """Resample a dense time-major array to a target sampling rate.

    Inputs:
    - signal: input array with shape [T, C].
    - source_hz: original sampling rate in Hz.
    - target_hz: destination sampling rate in Hz.

    Outputs:
    - Resampled array with shape [T_out, C].
    """
    if int(round(source_hz)) == int(target_hz):
        return signal.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("Resampling to a new sampling rate requires scipy. Install with `pip install -e .[physio]`.") from exc
    src = int(round(source_hz))
    tgt = int(target_hz)
    g = gcd(src, tgt)
    up = tgt // g
    down = src // g
    return resample_poly(signal, up=up, down=down, axis=0).astype(np.float32, copy=False)
