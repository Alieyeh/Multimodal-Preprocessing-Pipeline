"""PAMAP2 preprocessing helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_PAMAP2_WRIST_CANDIDATES = {
    "acc_x": ["acc_x", "hand_acc_x", "wrist_acc_x", "IMU_hand_acc_x_16g"],
    "acc_y": ["acc_y", "hand_acc_y", "wrist_acc_y", "IMU_hand_acc_y_16g"],
    "acc_z": ["acc_z", "hand_acc_z", "wrist_acc_z", "IMU_hand_acc_z_16g"],
    "gyro_x": ["gyro_x", "hand_gyro_x", "wrist_gyro_x", "IMU_hand_gyro_x"],
    "gyro_y": ["gyro_y", "hand_gyro_y", "wrist_gyro_y", "IMU_hand_gyro_y"],
    "gyro_z": ["gyro_z", "hand_gyro_z", "wrist_gyro_z", "IMU_hand_gyro_z"],
    "label": ["label", "activity_id", "activityID"],
    "time": ["time", "timestamp", "ts"],
}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available candidate column name."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def parse_pamap2_to_interim(path: str | Path, target_columns: list[str]) -> pd.DataFrame:
    """Parse a PAMAP2 source file into shared interim columns.

    Inputs:
    - path: source file path.
    - target_columns: shared HAR channel schema.

    Outputs:
    - DataFrame containing the requested channels and a `label` column.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#", header=None)

    if set(target_columns).issubset(df.columns) and "label" in df.columns:
        cols = target_columns + ["label"]
        if "time" in df.columns:
            cols.append("time")
        return df[cols].copy()

    if "label" not in df.columns and df.shape[1] >= 30:
        out = pd.DataFrame(
            {
                "time": df.iloc[:, 0],
                "label": df.iloc[:, 1],
                "acc_x": df.iloc[:, 21],
                "acc_y": df.iloc[:, 22],
                "acc_z": df.iloc[:, 23],
                "gyro_x": df.iloc[:, 27],
                "gyro_y": df.iloc[:, 28],
                "gyro_z": df.iloc[:, 29],
            }
        )
        return out[target_columns + ["label", "time"]].copy()

    mapping = {}
    for target, candidates in _PAMAP2_WRIST_CANDIDATES.items():
        picked = _pick_column(df, candidates)
        if picked is not None:
            mapping[target] = picked
    missing = [c for c in target_columns + ["label"] if c not in mapping]
    if missing:
        raise ValueError(f"PAMAP2 interim parse missing columns: {missing}")
    cols = [mapping[c] for c in [*target_columns, "label"]]
    if "time" in mapping:
        cols.append(mapping["time"])
    return df[cols].rename(columns={v: k for k, v in mapping.items()}).copy()
