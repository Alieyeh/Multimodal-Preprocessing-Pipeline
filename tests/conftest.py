from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml


def _har_df(n=400):
    t = np.arange(n)
    return pd.DataFrame({
        "acc_x": np.sin(t/10),
        "acc_y": np.cos(t/10),
        "acc_z": np.sin(t/7),
        "gyro_x": np.cos(t/9),
        "gyro_y": np.sin(t/8),
        "gyro_z": np.cos(t/6),
        "label": np.where((t // 50) % 2 == 0, "walk", "sit"),
    })


def make_fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path
    for d in ["data/raw/pamap2", "data/raw/wisdm", "data/raw/mhealth", "data/raw/eegmmidb", "data/raw/ptbxl", "reports", "submission_sample"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Fixture repo\n", encoding="utf-8")
    (root / "preprocessing_plan.md").write_text("# Fixture plan\n", encoding="utf-8")

    for dataset in ["pamap2", "wisdm", "mhealth"]:
        _har_df().to_csv(root / f"data/raw/{dataset}/{dataset}_subject1.csv", index=False)

    eeg_rows = 1000
    eeg = pd.DataFrame({f"ch_{i}": np.random.randn(eeg_rows) for i in range(64)})
    eeg["event_onset"] = np.nan
    eeg["event_code"] = pd.Series([None] * eeg_rows, dtype="object")
    eeg.loc[10, ["event_onset", "event_code"]] = [10, "T1"]
    eeg.loc[200, ["event_onset", "event_code"]] = [200, "T2"]
    eeg.to_csv(root / "data/raw/eegmmidb/S001_run4.csv", index=False)

    ecg_rows = 1000
    ecg = pd.DataFrame({f"lead_{i}": np.random.randn(ecg_rows) for i in range(12)})
    ecg["patient_id"] = "p1"
    ecg["record_id"] = "r1"
    ecg["label"] = "NORM"
    ecg["strat_fold"] = 10
    ecg.to_csv(root / "data/raw/ptbxl/record1.csv", index=False)

    cfg = {
        "runtime": {"preprocess_mode": "best-effort"},
        "datasets": {
            "pamap2": {"required": True},
            "wisdm": {"required": True},
            "mhealth": {"required": False},
            "eegmmidb": {"required": True},
            "ptbxl": {"required": True},
        },
        "paths": {
            "raw_dir": str(root / "data/raw"),
            "interim_dir": str(root / "data/interim"),
            "processed_dir": str(root / "data/processed"),
            "reports_dir": str(root / "reports"),
            "submission_sample_dir": str(root / "submission_sample"),
        },
        "har": {
            "target_hz": 20,
            "channel_schema": ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"],
            "null_labels": [0, "null", "transient"],
            "label_schema_name": "har_unified_v1",
            "label_maps": {
                "pamap2": {"walk": "walking", "sit": "sitting"},
                "wisdm": {"walk": "walking", "sit": "sitting"},
                "mhealth": {"walk": "walking", "sit": "sitting"},
            },
        },
        "pamap2": {"enabled": True, "source_sampling_rate_hz": 20},
        "wisdm": {"enabled": True, "source_sampling_rate_hz": 20},
        "mhealth": {"enabled": True, "source_sampling_rate_hz": 20},
        "eeg": {"enabled": True, "event_codes": ["T1", "T2"], "sampling_rate_hz": 160, "window_seconds": 4},
        "ecg": {"enabled": True, "sampling_rate_hz": 100, "holdout_fold": 10},
    }
    with (root / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    (root / "reports" / "download_manifest.json").write_text(
        json.dumps(
            [
                {"dataset_name": "pamap2", "status": "downloaded"},
                {"dataset_name": "wisdm", "status": "downloaded"},
                {"dataset_name": "mhealth", "status": "downloaded"},
                {"dataset_name": "eegmmidb", "status": "downloaded"},
                {"dataset_name": "ptbxl", "status": "downloaded"},
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "reports" / "setup_summary.md").write_text("# Setup summary\n", encoding="utf-8")
    (root / "reports" / "resource_estimate.md").write_text("# Resource estimate\n", encoding="utf-8")
    (root / "reports" / "reproducibility_context.json").write_text(
        json.dumps(
            {
                "config_path": str(root / "config.yaml"),
                "config_sha256": hashlib.sha256((root / "config.yaml").read_bytes()).hexdigest(),
                "python_version": "3.14.0",
                "platform": "fixture-platform",
                "completed_stages": ["setup"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root
