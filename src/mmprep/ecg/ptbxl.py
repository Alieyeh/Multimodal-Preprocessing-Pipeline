"""ECG preprocessing helpers for fixed-length PTB-XL records."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _stringify_identifier(value: object) -> str:
    """Preserve integer-like identifiers without a trailing `.0`."""
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def parse_ptbxl_csv_to_interim(path: str | Path) -> tuple[np.ndarray, dict]:
    """Parse a lightweight PTB-XL CSV fixture into a record tensor and metadata.

    Expected input format:
    - 12 columns named lead_0 .. lead_11.
    - Constant metadata columns: patient_id, record_id, label, strat_fold.

    Outputs:
    - Record tensor shaped [C, T].
    - Metadata dictionary.
    """
    df = pd.read_csv(path)
    lead_cols = [c for c in df.columns if c.startswith("lead_")]
    if len(lead_cols) != 12:
        raise ValueError("Expected 12 lead columns")
    meta = {
        "patient_id": _stringify_identifier(df["patient_id"].iloc[0]),
        "record_id": _stringify_identifier(df["record_id"].iloc[0]),
        "label": str(df["label"].iloc[0]),
        "strat_fold": int(df["strat_fold"].iloc[0]),
        "sampling_rate_hz": 100,
        "lead_names": DEFAULT_LEAD_ORDER,
    }
    return df[lead_cols].to_numpy(dtype=np.float32).T, meta


def _primary_scp_label(raw_codes: str) -> str:
    """Select the highest-weight SCP code from the PTB-XL metadata field."""
    parsed = ast.literal_eval(raw_codes) if isinstance(raw_codes, str) else raw_codes
    if not parsed:
        return "UNKNOWN"
    return sorted(parsed.items(), key=lambda kv: (-float(kv[1]), kv[0]))[0][0]


def load_ptbxl_metadata(base_dir: str | Path, target_sampling_rate_hz: int = 100) -> list[dict]:
    """Load PTB-XL metadata rows pointing to waveform records.

    Inputs:
    - base_dir: PTB-XL root directory containing `ptbxl_database.csv`.
    - target_sampling_rate_hz: 100 or 500 Hz choice.

    Outputs:
    - List of record descriptors with waveform paths and metadata.
    """
    base_dir = Path(base_dir)
    csv_path = base_dir / "ptbxl_database.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing PTB-XL metadata table: {csv_path}")
    meta = pd.read_csv(csv_path)
    file_col = "filename_lr" if int(target_sampling_rate_hz) == 100 else "filename_hr"
    rows = []
    for _, row in meta.iterrows():
        rows.append({
            "patient_id": _stringify_identifier(row["patient_id"]),
            "record_id": _stringify_identifier(row["ecg_id"]),
            "label": _primary_scp_label(row["scp_codes"]),
            "strat_fold": int(row["strat_fold"]),
            "waveform_stem": str(base_dir / row[file_col]),
            "sampling_rate_hz": int(target_sampling_rate_hz),
            "lead_names": DEFAULT_LEAD_ORDER,
        })
    return rows


def load_ptbxl_record(record_stem: str | Path) -> np.ndarray:
    """Load a real PTB-XL waveform record via WFDB.

    Outputs:
    - Record tensor shaped [C, T].
    """
    try:
        import wfdb
    except Exception as exc:  # pragma: no cover - requires optional dependency
        raise RuntimeError("Loading real PTB-XL records requires `wfdb`. Install with `pip install -e .[physio]`.") from exc

    signal, fields = wfdb.rdsamp(str(record_stem))
    return signal.T.astype(np.float32)


def record_to_processed(
    record: np.ndarray,
    meta: dict,
    sampling_rate_hz: int,
    holdout_fold: int,
    normalize_per_record: bool = True,
    remove_per_lead_mean: bool = True,
) -> tuple[np.ndarray, dict]:
    """Convert one PTB-XL record to final processed format.

    Inputs:
    - record: ECG waveform array [C, T].
    - meta: metadata dictionary.
    - sampling_rate_hz: selected dataset rate.
    - holdout_fold: fold reserved for final test split.
    - normalize_per_record: whether to z-score each lead within a record.
    - remove_per_lead_mean: whether to remove per-lead DC offset before z-scoring.

    Outputs:
    - Fixed-shape batch array [1, C, T].
    - Metadata row.
    """
    x = record.astype(np.float32, copy=True)
    qc_flags: list[str] = []
    if remove_per_lead_mean:
        x = x - x.mean(axis=1, keepdims=True)
    if normalize_per_record:
        sigma = x.std(axis=1, keepdims=True)
        sigma[sigma < 1e-6] = 1.0
        x = x / sigma
    if not np.isfinite(x).all():
        qc_flags.append("non_finite")
    split = "test" if int(meta["strat_fold"]) == int(holdout_fold) else "train"
    cv_fold = None if split == "test" else int(meta["strat_fold"])
    row = {
        "sample_id": f"ptbxl_{meta['record_id']}",
        "dataset_name": "ptbxl",
        "modality": "ecg",
        "subject_or_patient_id": meta["patient_id"],
        "source_file_or_record": meta["record_id"],
        "split": split,
        "label_or_event": meta["label"],
        "sampling_rate_hz": sampling_rate_hz,
        "n_channels": x.shape[0],
        "n_samples": x.shape[1],
        "channel_schema": "|".join(meta.get("lead_names", DEFAULT_LEAD_ORDER)),
        "lead_names": "|".join(meta.get("lead_names", DEFAULT_LEAD_ORDER)),
        "qc_flags": "|".join(qc_flags),
        "strat_fold": meta["strat_fold"],
        "cv_fold": cv_fold,
        # Filled in after the full PTB-XL table is assembled so it reflects the
        # actual train/test patient partition rather than an assumption.
        "patient_safe_split": None,
        "window_seconds": x.shape[1] / float(sampling_rate_hz),
    }
    return x[np.newaxis, ...].astype(np.float32), row
