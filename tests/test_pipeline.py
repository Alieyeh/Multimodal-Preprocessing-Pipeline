from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np

from mmprep.pipeline import run_preprocess
from mmprep.validation import validate_processed
from mmprep.common.reproducibility import update_reproducibility_context
from tests.conftest import make_fixture_repo
from mmprep.pipeline import _discover_har_files


def test_pipeline_creates_interim_and_processed(tmp_path):
    root = make_fixture_repo(tmp_path)
    assert run_preprocess(root / "config.yaml") == 0
    assert (root / "data/interim/har/pamap2/pamap2_subject1_interim.csv").exists()
    assert (root / "data/interim/eeg/S001_run4_summary.json").exists()
    assert (root / "data/interim/ecg/record1_summary.json").exists()
    assert (root / "data/processed/har/pamap2_pretrain.npz").exists()
    assert (root / "reports/processed_manifest.json").exists()


def test_validation_passes_on_fixture_repo(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    update_reproducibility_context(root / "reports", root / "config.yaml", "validate")
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample")
    assert ok
    assert "PASS" in report
    assert "Resource awareness" in report


def test_pipeline_resume_skips_existing_bundles(tmp_path):
    root = make_fixture_repo(tmp_path)
    assert run_preprocess(root / "config.yaml") == 0
    assert run_preprocess(root / "config.yaml", resume=True) == 0


def test_validation_fails_when_no_processed_arrays_exist(tmp_path):
    root = make_fixture_repo(tmp_path)
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample")
    assert not ok
    assert "no processed array bundles were found" in report


def test_validation_fails_when_manifest_does_not_cover_processed_files(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    manifest_path = root / "reports/processed_manifest.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest = [entry for entry in manifest if not str(entry.get("path", "")).endswith("pamap2_pretrain.csv")]
    manifest_path.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    ok, report = validate_processed(root / "data/processed", manifest_path, root / "submission_sample")
    assert not ok
    assert "unmanifested_files=1" in report


def test_validation_fails_when_required_submission_docs_are_missing(tmp_path):
    root = make_fixture_repo(tmp_path)
    run_preprocess(root / "config.yaml")
    (root / "README.md").unlink()
    ok, report = validate_processed(root / "data/processed", root / "reports/processed_manifest.json", root / "submission_sample")
    assert not ok
    assert "missing_docs=1" in report


def test_sample_pack_generator_writes_modality_subfolders(tmp_path):
    root = make_fixture_repo(tmp_path)
    assert run_preprocess(root / "config.yaml") == 0
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/make_submission_sample.py"), "--processed-dir", str(root / "data/processed"), "--out-dir", str(root / "submission_sample"), "--n", "10"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert any((root / "submission_sample").glob("har/*/*_sample.csv"))
    assert any((root / "submission_sample").glob("eeg/*/*_sample.csv"))
    assert any((root / "submission_sample").glob("ecg/*/*_sample.csv"))
    assert (root / "submission_sample" / "sample_pack_manifest.json").exists()


def test_mhealth_file_discovery_deduplicates_duplicate_archive_wrapper_paths(tmp_path):
    dataset_raw = tmp_path / "data" / "raw" / "mhealth"
    a = dataset_raw / "MHEALTHDATASET"
    b = dataset_raw / "mhealth%2Bdataset" / "MHEALTHDATASET"
    a.mkdir(parents=True, exist_ok=True)
    b.mkdir(parents=True, exist_ok=True)
    (a / "mHealth_subject1.log").write_text("a", encoding="utf-8")
    (b / "mHealth_subject1.log").write_text("b", encoding="utf-8")
    files = _discover_har_files("mhealth", dataset_raw)
    assert len(files) == 1
    assert files[0].parent == a


def test_empty_ecg_bundle_uses_configured_sampling_rate(tmp_path):
    root = make_fixture_repo(tmp_path)
    ptbxl_dir = root / "data/raw/ptbxl"
    for csv_path in ptbxl_dir.glob("*.csv"):
        csv_path.unlink()
    (ptbxl_dir / "ptbxl_database.csv").write_text(
        "patient_id,ecg_id,scp_codes,strat_fold,filename_lr,filename_hr\n",
        encoding="utf-8",
    )
    import yaml
    cfg_path = root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["ecg"]["sampling_rate_hz"] = 500
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert run_preprocess(cfg_path) == 0
    arr = np.load(root / "data/processed/ecg/ptbxl_records.npz")["X"]
    assert arr.shape == (0, 12, 5000)
