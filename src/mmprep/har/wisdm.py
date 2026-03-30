"""WISDM preprocessing helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


_WISDM_CANDIDATES = {
    "acc_x": ["acc_x", "watch_acc_x", "x_acc", "x-axis_accel", "x-axis"],
    "acc_y": ["acc_y", "watch_acc_y", "y_acc", "y-axis_accel", "y-axis"],
    "acc_z": ["acc_z", "watch_acc_z", "z_acc", "z-axis_accel", "z-axis"],
    "gyro_x": ["gyro_x", "watch_gyro_x", "x_gyro"],
    "gyro_y": ["gyro_y", "watch_gyro_y", "y_gyro"],
    "gyro_z": ["gyro_z", "watch_gyro_z", "z_gyro"],
    "label": ["label", "activity", "activity_label"],
    "time": ["time", "timestamp", "unix_timestamp"],
}

_WISDM_ACTIVITY_MAP = {
    "A": "walking",
    "B": "jogging",
    "C": "stairs",
    "D": "sitting",
    "E": "standing",
    "F": "typing",
    "G": "teeth",
    "H": "soup",
    "I": "chips",
    "J": "pasta",
    "K": "drinking",
    "L": "sandwich",
    "M": "kicking",
    "O": "catch",
    "P": "dribbling",
    "Q": "writing",
    "R": "clapping",
    "S": "folding",
}


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first available candidate column name."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _load_wisdm_sensor_file(path: Path) -> pd.DataFrame:
    """Load one raw WISDM sensor file with subject, activity, timestamp, x, y, z."""
    df = pd.read_csv(
        path,
        header=None,
        names=["subject_id", "activity_code", "timestamp", "x", "y", "z"],
        sep=",",
        engine="python",
    )
    df["z"] = df["z"].astype(str).str.rstrip(";")
    df["activity_code"] = df["activity_code"].astype(str).str.strip()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") / 1e9
    for col in ["x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)


def _load_wisdm_watch_pair(accel_path: Path) -> pd.DataFrame:
    """Load and merge paired WISDM watch accel and gyro files."""
    gyro_path = accel_path.parent.parent / "gyro" / accel_path.name.replace("_accel_watch", "_gyro_watch")
    if not gyro_path.exists():
        raise FileNotFoundError(f"Missing paired WISDM gyro file for {accel_path.name}")
    acc = _load_wisdm_sensor_file(accel_path).rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    gyro = _load_wisdm_sensor_file(gyro_path).rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    acc = acc.sort_values(["subject_id", "activity_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    gyro = gyro.sort_values(["subject_id", "activity_code", "timestamp"], kind="mergesort").reset_index(drop=True)
    merged_groups: list[pd.DataFrame] = []
    gyro_cols = ["timestamp", "gyro_x", "gyro_y", "gyro_z"]
    for (subject_id, activity_code), acc_group in acc.groupby(["subject_id", "activity_code"], sort=False):
        gyro_group = gyro[
            (gyro["subject_id"] == subject_id) & (gyro["activity_code"] == activity_code)
        ][gyro_cols]
        if gyro_group.empty:
            continue
        merged_groups.append(
            pd.merge_asof(
                acc_group.sort_values("timestamp", kind="mergesort"),
                gyro_group.sort_values("timestamp", kind="mergesort"),
                on="timestamp",
                direction="nearest",
                tolerance=1.0 / 40.0,
            )
        )
    merged = pd.concat(merged_groups, ignore_index=True) if merged_groups else acc.iloc[0:0].copy()
    merged = merged.dropna(subset=["gyro_x", "gyro_y", "gyro_z"]).reset_index(drop=True)
    min_expected = max(1, int(0.8 * min(len(acc), len(gyro))))
    if len(merged) < min_expected:
        raise ValueError(
            f"WISDM watch accel/gyro alignment retained only {len(merged)} paired rows "
            f"from accel={len(acc)} and gyro={len(gyro)}; timestamps appear too misaligned."
        )
    merged["label"] = merged["activity_code"].map(_WISDM_ACTIVITY_MAP).fillna(merged["activity_code"].astype(str))
    merged["time"] = merged["timestamp"]
    return merged[["subject_id", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label", "time"]].copy()


def parse_wisdm_to_interim(path: str | Path, target_columns: list[str]) -> pd.DataFrame:
    """Parse a WISDM source file into shared interim columns."""
    path = Path(path)
    if path.suffix.lower() == ".txt" and "_accel_watch" in path.name:
        return _load_wisdm_watch_pair(path)

    df = pd.read_csv(path)
    if set(target_columns).issubset(df.columns) and "label" in df.columns:
        cols = target_columns + ["label"]
        if "subject_id" in df.columns:
            cols.insert(0, "subject_id")
        if "time" in df.columns:
            cols.append("time")
        return df[cols].copy()

    mapping = {}
    for target, candidates in _WISDM_CANDIDATES.items():
        picked = _pick_column(df, candidates)
        if picked is not None:
            mapping[target] = picked
    missing = [c for c in target_columns + ["label"] if c not in mapping]
    if missing:
        raise ValueError(f"WISDM interim parse missing columns: {missing}")
    cols = [mapping[c] for c in [*target_columns, "label"]]
    if "time" in mapping:
        cols.append(mapping["time"])
    return df[cols].rename(columns={v: k for k, v in mapping.items()}).copy()
