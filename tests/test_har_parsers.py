from pathlib import Path

import pandas as pd

from mmprep.har.pamap2 import parse_pamap2_to_interim
from mmprep.har.wisdm import parse_wisdm_to_interim


def test_pamap2_parser_accepts_candidate_column_names(tmp_path: Path):
    df = pd.DataFrame(
        {
            "wrist_acc_x": [0.1, 0.2],
            "wrist_acc_y": [0.3, 0.4],
            "wrist_acc_z": [0.5, 0.6],
            "wrist_gyro_x": [0.7, 0.8],
            "wrist_gyro_y": [0.9, 1.0],
            "wrist_gyro_z": [1.1, 1.2],
            "activity_id": [4, 4],
            "timestamp": [0.0, 0.01],
        }
    )
    path = tmp_path / "pamap2.csv"
    df.to_csv(path, index=False)
    out = parse_pamap2_to_interim(path, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    assert set(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label"]).issubset(out.columns)


def test_wisdm_parser_accepts_candidate_column_names(tmp_path: Path):
    df = pd.DataFrame(
        {
            "watch_acc_x": [0.1, 0.2],
            "watch_acc_y": [0.3, 0.4],
            "watch_acc_z": [0.5, 0.6],
            "watch_gyro_x": [0.7, 0.8],
            "watch_gyro_y": [0.9, 1.0],
            "watch_gyro_z": [1.1, 1.2],
            "activity": ["walking", "walking"],
            "unix_timestamp": [0.0, 0.05],
        }
    )
    path = tmp_path / "wisdm.csv"
    df.to_csv(path, index=False)
    out = parse_wisdm_to_interim(path, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    assert set(["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label"]).issubset(out.columns)
