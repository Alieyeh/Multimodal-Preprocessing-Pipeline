"""EEG preprocessing helpers for EEGMMIDB event-aligned windows."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_eeg_csv_to_interim(path: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Parse a lightweight EEG CSV fixture into signal and event table.

    Expected input format:
    - CSV columns named `ch_*` for channels.
    - `event_onset` integer sample index column.
    - `event_code` event code column.

    Outputs:
    - Signal array shaped [T, C].
    - Event table with onset and event code.
    """
    df = pd.read_csv(path)
    channel_cols = [c for c in df.columns if c.startswith("ch_")]
    if not channel_cols:
        raise ValueError("No EEG channel columns found")
    signal = df[channel_cols].to_numpy(dtype=np.float32)
    events = df[["event_onset", "event_code"]].dropna().drop_duplicates().reset_index(drop=True)
    return signal, events


def parse_eeg_edf_to_interim(
    path: str | Path,
    bandpass_hz: tuple[float, float] | None = (1.0, 40.0),
    notch_hz: float | None = 60.0,
    rereference: str | None = "average",
) -> tuple[np.ndarray, pd.DataFrame, float, list[str]]:
    """Parse a real EEGMMIDB EDF+ file into signal and annotation table.

    Inputs:
    - path: EDF+ filepath.
    - bandpass_hz: optional (low, high) band-pass filter range.
    - notch_hz: optional line-noise notch frequency.
    - rereference: optional rereferencing mode; `average` is the default.

    Outputs:
    - Signal array shaped [T, C].
    - Annotation table with onset, duration, and event code.
    - Final sampling rate in Hz.
    - Channel names.
    """
    try:
        import mne
    except Exception as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError("Real EDF ingestion requires `mne`. Install with `pip install -e .[physio]`.") from exc

    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    if raw.info.get("nchan", 0) == 0:
        raise ValueError(f"No channels found in {path}")
    raw.pick("eeg")
    if rereference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if bandpass_hz is not None:
        low, high = bandpass_hz
        raw.filter(low, high, verbose="ERROR")
    if notch_hz is not None and notch_hz < (raw.info["sfreq"] / 2.0):
        raw.notch_filter(notch_hz, verbose="ERROR")

    signal = raw.get_data().T.astype(np.float32)
    events = pd.DataFrame(
        {
            "event_onset_seconds": raw.annotations.onset,
            "event_duration_seconds": raw.annotations.duration,
            "event_code": raw.annotations.description,
        }
    )
    events["event_onset"] = (events["event_onset_seconds"] * raw.info["sfreq"]).round().astype(int)
    return signal, events, float(raw.info["sfreq"]), list(raw.ch_names)


def infer_subject_run_from_name(name: str) -> tuple[str, str]:
    """Infer EEGMMIDB subject and run identifiers from a file stem or path name."""
    match = re.search(r"S?(\d{3}).*R(?:un)?0?(\d{1,2})", name, flags=re.IGNORECASE)
    if match:
        return match.group(1), str(int(match.group(2)))
    match = re.search(r"(\d{3}).*run[_-]?(\d{1,2})", name, flags=re.IGNORECASE)
    if match:
        return match.group(1), str(int(match.group(2)))
    return "unknown", "unknown"


def event_windows(
    signal: np.ndarray,
    events: pd.DataFrame,
    sampling_rate_hz: int,
    window_seconds: int,
    subject_id: str,
    run_id: str,
    source_name: str,
    channel_names: list[str] | None = None,
    normalize_per_window: bool = True,
    allowed_event_codes: set[str] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Create fixed event-aligned EEG windows.

    Inputs:
    - signal: array [T, C].
    - events: DataFrame with `event_onset` and `event_code`.
    - sampling_rate_hz: final sampling rate.
    - window_seconds: window duration in seconds.
    - subject_id: subject identifier.
    - run_id: EEG run identifier.
    - source_name: source file identifier.
    - channel_names: optional list of EEG channel names.
    - normalize_per_window: whether to z-score each channel within each window.

    Outputs:
    - Window tensor [N, C, T].
    - Metadata rows.
    """
    n = sampling_rate_hz * window_seconds
    xs = []
    rows = []
    schema = "|".join(channel_names) if channel_names else f"ch0..ch{signal.shape[1]-1}"
    allowed_codes = {str(code) for code in allowed_event_codes} if allowed_event_codes else None
    for i, row in events.iterrows():
        event_code = str(row["event_code"])
        if allowed_codes is not None and event_code not in allowed_codes:
            continue
        start = int(row["event_onset"])
        end = start + n
        if end > signal.shape[0]:
            continue
        window = signal[start:end].T.astype(np.float32, copy=False)
        qc_flags: list[str] = []
        if not np.isfinite(window).all():
            qc_flags.append("non_finite")
            continue
        if normalize_per_window:
            mu = window.mean(axis=1, keepdims=True)
            sigma = window.std(axis=1, keepdims=True)
            sigma[sigma < 1e-6] = 1.0
            window = ((window - mu) / sigma).astype(np.float32, copy=False)
        if np.abs(window).max() > 20:
            qc_flags.append("high_amplitude")
        xs.append(window)
        rows.append({
            "sample_id": f"eegmmidb_{subject_id}_{run_id}_{i:05d}",
            "dataset_name": "eegmmidb",
            "modality": "eeg",
            "subject_or_patient_id": subject_id,
            "source_file_or_record": source_name,
            "split": "unspecified",
            "label_or_event": event_code,
            "event_code": event_code,
            "sampling_rate_hz": sampling_rate_hz,
            "n_channels": signal.shape[1],
            "n_samples": n,
            "channel_schema": schema,
            "qc_flags": "|".join(qc_flags),
            "run_id": run_id,
            "event_onset": start,
            "event_onset_seconds": float(row.get("event_onset_seconds", start / float(sampling_rate_hz))),
        })
    if not xs:
        return np.empty((0, signal.shape[1], n), dtype=np.float32), []
    return np.stack(xs).astype(np.float32), rows
