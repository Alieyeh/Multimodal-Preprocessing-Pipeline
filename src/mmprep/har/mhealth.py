"""mHealth preprocessing helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_MHEALTH_CANDIDATES = {
    "acc_x": ["acc_x", "right_lower_arm_acc_x", "rlarm_acc_x"],
    "acc_y": ["acc_y", "right_lower_arm_acc_y", "rlarm_acc_y"],
    "acc_z": ["acc_z", "right_lower_arm_acc_z", "rlarm_acc_z"],
    "gyro_x": ["gyro_x", "right_lower_arm_gyro_x", "rlarm_gyro_x"],
    "gyro_y": ["gyro_y", "right_lower_arm_gyro_y", "rlarm_gyro_y"],
    "gyro_z": ["gyro_z", "right_lower_arm_gyro_z", "rlarm_gyro_z"],
    "label": ["label", "activity", "activity_id"],
    "time": ["time", "timestamp", "ts"],
}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available candidate column name."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _load_mhealth_log(path: Path) -> pd.DataFrame:
    """Load a raw mHealth subject log using the documented column order."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < 24:
        raise ValueError(f"Unexpected mHealth log width for {path.name}: {df.shape[1]}")
    out = pd.DataFrame(
        {
            "acc_x": df.iloc[:, 14],
            "acc_y": df.iloc[:, 15],
            "acc_z": df.iloc[:, 16],
            "gyro_x": df.iloc[:, 17],
            "gyro_y": df.iloc[:, 18],
            "gyro_z": df.iloc[:, 19],
            "label": df.iloc[:, 23],
        }
    )
    return out


def parse_mhealth_to_interim(path: str | Path, target_columns: list[str]) -> pd.DataFrame:
    """Parse an mHealth source file into shared interim columns."""
    path = Path(path)
    if path.suffix.lower() == ".log":
        return _load_mhealth_log(path)

    df = pd.read_csv(path)
    if set(target_columns).issubset(df.columns) and "label" in df.columns:
        cols = target_columns + ["label"]
        if "time" in df.columns:
            cols.append("time")
        return df[cols].copy()

    mapping = {}
    for target, candidates in _MHEALTH_CANDIDATES.items():
        picked = _pick_column(df, candidates)
        if picked is not None:
            mapping[target] = picked
    missing = [c for c in target_columns + ["label"] if c not in mapping]
    if missing:
        raise ValueError(f"mHealth interim parse missing columns: {missing}")
    cols = [mapping[c] for c in [*target_columns, "label"]]
    if "time" in mapping:
        cols.append(mapping["time"])
    return df[cols].rename(columns={v: k for k, v in mapping.items()}).copy()
