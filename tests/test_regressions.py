from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mmprep.pipeline import run_preprocess
from mmprep.pipeline import _write_compliance_outputs
from mmprep.validation import validate_processed
from mmprep.ecg.ptbxl import load_ptbxl_metadata
from mmprep.common.reproducibility import update_reproducibility_context
from mmprep.har.wisdm import parse_wisdm_to_interim
from setup_data import perform_setup
from tests.conftest import make_fixture_repo


def test_empty_har_outputs_keep_expected_window_lengths(tmp_path):
    root = make_fixture_repo(tmp_path)
    for dataset in ["pamap2", "wisdm", "mhealth"]:
        for path in (root / f"data/raw/{dataset}").glob("*.csv"):
            path.unlink()
    run_preprocess(root / "config.yaml")
    # HAR folders should not exist because missing datasets are skipped, but EEG shape should remain stable
    eeg = np.load(root / "data/processed/eeg/eegmmidb_events.npz")["X"]
    assert eeg.shape[-1] == 640


def test_setup_can_exclude_mhealth_and_writes_compliance(tmp_path):
    cfg = {
        "runtime": {"setup_mode": "best-effort", "http_timeout_seconds": 10, "retry_count": 1, "ptbxl_download_workers": 1},
        "datasets": {
            "pamap2": {"source_url": "https://example.com/pamap2.zip", "dataset_version": None, "required": True},
            "wisdm": {"source_url": "https://example.com/wisdm.zip", "dataset_version": None, "required": True},
            "mhealth": {"source_url": "https://example.com/mhealth.zip", "dataset_version": None, "required": False},
            "eegmmidb": {"source_url": "https://physionet.org/files/eegmmidb/1.0.0/", "dataset_version": "1.0.0", "required": True},
            "ptbxl": {"source_url": "https://physionet.org/files/ptb-xl/1.0.3/", "dataset_version": "1.0.3", "required": True},
        },
        "eeg": {"runs": [4, 8, 12], "subjects": [1]},
        "ecg": {"sampling_rate_hz": 100, "max_records": 1},
    }
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "default.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    manifest = perform_setup(tmp_path, include_mhealth=False, simulate_downloads=True, config_path="configs/default.yaml")
    assert all(item["dataset_name"] != "mhealth" for item in manifest)
    assert (tmp_path / "reports/compliance_checklist.json").exists()


def test_sample_script_writes_bounded_outputs(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / 'scripts' / 'make_submission_sample.py'), '--processed-dir', str(root / 'data/processed'), '--out-dir', str(root / 'submission_sample'), '--n', '1', '--seed', '7'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    csvs = sorted((root / 'submission_sample').glob('**/*.csv'))
    assert csvs
    for csv in csvs:
        import pandas as pd
        assert len(pd.read_csv(csv)) <= 1


def test_sample_script_targets_total_rows_per_dataset(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / 'scripts' / 'make_submission_sample.py'),
            '--processed-dir',
            str(root / 'data/processed'),
            '--out-dir',
            str(root / 'submission_sample'),
            '--n',
            '10',
            '--clean',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = __import__("json").loads((root / "submission_sample" / "sample_pack_manifest.json").read_text(encoding="utf-8"))
    per_dataset: dict[str, int] = {}
    expected_totals: dict[str, int] = {}
    for item in manifest["files"]:
        per_dataset[item["dataset_name"]] = per_dataset.get(item["dataset_name"], 0) + int(item["expected_rows"])
        expected_totals[item["dataset_name"]] = int(item["dataset_target_rows"])
    assert per_dataset == expected_totals


def test_har_supervised_outputs_preserve_unified_and_original_labels(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    df = __import__("pandas").read_csv(root / "data/processed/har/pamap2_supervised.csv")
    assert {"label_schema_name", "original_label", "null_label_policy"}.issubset(df.columns)
    assert set(df["label_or_event"].dropna().unique()) <= {"walking", "sitting"}


def test_validation_fails_on_ptbxl_patient_overlap(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    csv_path = root / "data/processed/ecg/ptbxl_records.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    extra = df.iloc[[0]].copy()
    extra["sample_id"] = "ptbxl_overlap_copy"
    extra["split"] = "train"
    extra["cv_fold"] = 1
    extra["patient_safe_split"] = True
    pd.concat([df, extra], ignore_index=True).to_csv(csv_path, index=False)
    arr_path = root / "data/processed/ecg/ptbxl_records.npz"
    arr = np.load(arr_path, allow_pickle=False)["X"]
    np.savez_compressed(arr_path, X=np.concatenate([arr, arr], axis=0))
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample")
    assert not ok
    assert "patient leakage detected" in report


def test_wisdm_watch_pairing_tolerates_small_timestamp_jitter(tmp_path):
    base = tmp_path / "wisdm"
    accel_dir = base / "watch" / "accel"
    gyro_dir = base / "watch" / "gyro"
    accel_dir.mkdir(parents=True)
    gyro_dir.mkdir(parents=True)
    accel = pd.DataFrame(
        [
            [1, "A", 0, 0.1, 0.2, 0.3],
            [1, "A", 50_000_000, 0.4, 0.5, 0.6],
            [1, "A", 100_000_000, 0.7, 0.8, 0.9],
        ]
    )
    gyro = pd.DataFrame(
        [
            [1, "A", 1_000_000, 1.1, 1.2, 1.3],
            [1, "A", 49_000_000, 1.4, 1.5, 1.6],
            [1, "A", 101_000_000, 1.7, 1.8, 1.9],
        ]
    )
    accel_path = accel_dir / "data_1600_accel_watch.txt"
    gyro_path = gyro_dir / "data_1600_gyro_watch.txt"
    accel.to_csv(accel_path, index=False, header=False)
    gyro.to_csv(gyro_path, index=False, header=False)
    out = parse_wisdm_to_interim(accel_path, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    assert len(out) == 3
    assert list(out.columns) == ["subject_id", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "label", "time"]


def test_wisdm_watch_pairing_handles_multiple_activity_groups(tmp_path):
    base = tmp_path / "wisdm"
    accel_dir = base / "watch" / "accel"
    gyro_dir = base / "watch" / "gyro"
    accel_dir.mkdir(parents=True)
    gyro_dir.mkdir(parents=True)
    accel = pd.DataFrame(
        [
            [1, "A", 0, 0.1, 0.2, 0.3],
            [1, "A", 50_000_000, 0.4, 0.5, 0.6],
            [1, "B", 0, 0.7, 0.8, 0.9],
            [1, "B", 50_000_000, 1.0, 1.1, 1.2],
        ]
    )
    gyro = pd.DataFrame(
        [
            [1, "A", 1_000_000, 1.1, 1.2, 1.3],
            [1, "A", 49_000_000, 1.4, 1.5, 1.6],
            [1, "B", 1_000_000, 1.7, 1.8, 1.9],
            [1, "B", 49_000_000, 2.0, 2.1, 2.2],
        ]
    )
    accel_path = accel_dir / "data_1600_accel_watch.txt"
    gyro_path = gyro_dir / "data_1600_gyro_watch.txt"
    accel.to_csv(accel_path, index=False, header=False)
    gyro.to_csv(gyro_path, index=False, header=False)
    out = parse_wisdm_to_interim(accel_path, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
    assert len(out) == 4
    assert "subject_id" in out.columns
    assert set(out["subject_id"]) == {1}
    assert set(out["label"]) == {"walking", "jogging"}


def test_wisdm_processed_outputs_preserve_real_subject_ids(tmp_path):
    root = make_fixture_repo(tmp_path)
    wisdm_path = root / "data" / "raw" / "wisdm" / "wisdm_subjects.csv"
    df = pd.DataFrame(
        {
            "subject_id": ["1"] * 220 + ["2"] * 220,
            "acc_x": np.sin(np.arange(440) / 10),
            "acc_y": np.cos(np.arange(440) / 10),
            "acc_z": np.sin(np.arange(440) / 7),
            "gyro_x": np.cos(np.arange(440) / 9),
            "gyro_y": np.sin(np.arange(440) / 8),
            "gyro_z": np.cos(np.arange(440) / 6),
            "label": ["walk"] * 440,
            "time": np.arange(440) / 20.0,
        }
    )
    df.to_csv(wisdm_path, index=False)
    run_preprocess(root / "config.yaml")
    meta = pd.read_csv(root / "data" / "processed" / "har" / "wisdm_pretrain.csv")
    assert {"1", "2"}.issubset(set(meta["subject_or_patient_id"].astype(str)))


def test_eeg_and_ecg_outputs_include_explicit_brief_fields(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    eeg = pd.read_csv(root / "data" / "processed" / "eeg" / "eegmmidb_events.csv")
    ecg = pd.read_csv(root / "data" / "processed" / "ecg" / "ptbxl_records.csv")
    assert "event_code" in eeg.columns
    assert (eeg["event_code"].astype(str) == eeg["label_or_event"].astype(str)).all()
    assert "lead_names" in ecg.columns
    assert (ecg["lead_names"].astype(str) == ecg["channel_schema"].astype(str)).all()


def test_validate_outputs_reports_recursive_sample_count(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    repo = Path(__file__).resolve().parents[1]
    expected_count = len(list((root / "data/processed").glob("**/*.csv")))
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "make_submission_sample.py"), "--processed-dir", str(root / "data/processed"), "--out-dir", str(root / "submission_sample"), "--n", "2"],
        capture_output=True,
        text=True,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "validate_outputs.py"),
            "--processed-dir",
            str(root / "data/processed"),
            "--manifest-path",
            str(root / "reports/processed_manifest.json"),
            "--report-path",
            str(root / "reports/validation_report.md"),
            "--sample-dir",
            str(root / "submission_sample"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"Submission sample metadata files checked: {expected_count}" in result.stdout


def test_validation_uses_configured_eeg_runs_and_labels(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["eeg"]["runs"] = [8]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_preprocess(cfg_path)
    ok, report = validate_processed(
        root / "data/processed",
        root / "reports/processed_manifest.json",
        root / "submission_sample",
        config_path=cfg_path,
    )
    assert not ok
    assert "configured set: ['8']" in report


def test_har_standardize_per_window_is_config_driven(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["har"]["standardize_per_window"] = True
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_preprocess(cfg_path)
    arr = np.load(root / "data/processed/har/pamap2_pretrain.npz", allow_pickle=False)["X"]
    assert arr.shape[0] > 0
    means = arr.mean(axis=2)
    stds = arr.std(axis=2)
    assert np.allclose(means, 0.0, atol=1e-4)
    assert np.allclose(stds, 1.0, atol=1e-3)


def test_eeg_include_t0_adds_rest_windows(tmp_path):
    root = make_fixture_repo(tmp_path)
    eeg_path = root / "data/raw/eegmmidb/S001_run4.csv"
    df = pd.read_csv(eeg_path)
    df.loc[300, ["event_onset", "event_code"]] = [300, "T0"]
    df.to_csv(eeg_path, index=False)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["eeg"]["include_t0"] = True
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_preprocess(cfg_path)
    meta = pd.read_csv(root / "data/processed/eeg/eegmmidb_events.csv")
    assert "T0" in set(meta["label_or_event"].astype(str))


def test_validation_fails_when_har_supervised_contains_unexpected_label(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    csv_path = root / "data/processed/har/pamap2_supervised.csv"
    df = pd.read_csv(csv_path)
    df.loc[0, "label_or_event"] = "teleporting"
    df.to_csv(csv_path, index=False)
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample")
    assert not ok
    assert "unexpected labels" in report


def test_compliance_checklist_buckets_manifest_entries_by_dataset(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        {"path": "data/processed/har/pamap2_pretrain.npz"},
        {"path": "data/processed/har/pamap2_pretrain.csv"},
        {"path": "data/processed/eeg/eegmmidb_events.npz"},
        {"path": "data/processed/ecg/ptbxl_records.csv"},
    ]
    _write_compliance_outputs(reports_dir, manifest, [], {"runtime": {"preprocess_mode": "best-effort"}})
    payload = __import__("json").loads((reports_dir / "compliance_checklist.json").read_text(encoding="utf-8"))
    assert payload["dataset_outputs"] == {"pamap2": 2, "eegmmidb": 1, "ptbxl": 1}


def test_ptbxl_processed_metadata_preserves_integer_like_identifiers_without_decimal_suffix(tmp_path):
    base = tmp_path / "ptbxl"
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "patient_id": [15709],
            "ecg_id": [1],
            "scp_codes": ["{'NORM': 100.0}"],
            "strat_fold": [3],
            "filename_lr": ["records100/00000/00001_lr"],
            "filename_hr": ["records500/00000/00001_hr"],
        }
    ).to_csv(base / "ptbxl_database.csv", index=False)
    rows = load_ptbxl_metadata(base, target_sampling_rate_hz=100)
    assert rows[0]["patient_id"] == "15709"
    assert rows[0]["record_id"] == "1"


def test_validation_respects_configured_har_label_space(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["har"]["label_maps"] = {
        "pamap2": {"walk": "amble", "sit": "perch"},
        "wisdm": {"walk": "amble", "sit": "perch"},
        "mhealth": {"walk": "amble", "sit": "perch"},
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_preprocess(cfg_path)
    update_reproducibility_context(root / "reports", cfg_path, "validate")
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample", config_path=cfg_path)
    assert ok
    assert "unexpected labels: none" in report


def test_validation_respects_configured_har_schema_name(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["har"]["label_schema_name"] = "har_project_custom_v2"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_preprocess(cfg_path)
    update_reproducibility_context(root / "reports", cfg_path, "validate")
    ok, report = validate_processed(
        root / "data/processed",
        root / "reports/processed_manifest.json",
        root / "submission_sample",
        config_path=cfg_path,
    )
    assert ok
    assert "expected: har_project_custom_v2" in report
    df = pd.read_csv(root / "data/processed/har/pamap2_supervised.csv")
    assert set(df["label_schema_name"]) == {"har_project_custom_v2"}


def test_validation_fails_when_reproducibility_context_is_missing_validate_stage(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    context_path = root / "reports" / "reproducibility_context.json"
    context = __import__("json").loads(context_path.read_text(encoding="utf-8"))
    context["completed_stages"] = ["setup", "preprocess"]
    context_path.write_text(__import__("json").dumps(context, indent=2), encoding="utf-8")
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample")
    assert not ok
    assert "missing_stages=['validate']" in report


def test_validation_fails_when_test_rows_do_not_use_configured_holdout_fold(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["ecg"]["holdout_fold"] = 3
    cfg["ecg"]["train_folds"] = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    run_preprocess(cfg_path)
    update_reproducibility_context(root / "reports", cfg_path, "validate")
    csv_path = root / "data/processed/ecg/ptbxl_records.csv"
    df = pd.read_csv(csv_path)
    df.loc[:, "split"] = "test"
    df.loc[:, "strat_fold"] = 10
    df = df.assign(cv_fold=pd.Series([np.nan] * len(df), dtype="Float64"))
    df.to_csv(csv_path, index=False)
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample", config_path=cfg_path)
    assert not ok
    assert "holdout fold check failed" in report


def test_validation_fails_when_train_rows_use_fold_outside_configured_train_folds(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["ecg"]["holdout_fold"] = 10
    cfg["ecg"]["train_folds"] = [1, 2, 3]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ptbxl_csv = root / "data/raw/ptbxl/record1.csv"
    df = pd.read_csv(ptbxl_csv)
    df["strat_fold"] = 4
    df.to_csv(ptbxl_csv, index=False)
    run_preprocess(cfg_path)
    update_reproducibility_context(root / "reports", cfg_path, "validate")
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample", config_path=cfg_path)
    assert not ok
    assert "train fold check failed" in report


def test_validation_reports_configured_ecg_holdout_fold(tmp_path):
    root = make_fixture_repo(tmp_path)
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["ecg"]["holdout_fold"] = 3
    cfg["ecg"]["train_folds"] = [1, 2, 4, 5, 6, 7, 8, 9, 10]
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    ptbxl_csv = root / "data/raw/ptbxl/record1.csv"
    df = pd.read_csv(ptbxl_csv)
    df["strat_fold"] = 3
    df.to_csv(ptbxl_csv, index=False)
    run_preprocess(cfg_path)
    update_reproducibility_context(root / "reports", cfg_path, "validate")
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample", config_path=cfg_path)
    assert ok
    assert "| test | 1 | 100.00% | 3 |" in report
