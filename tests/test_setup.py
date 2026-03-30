import json
from pathlib import Path
import zipfile

import yaml

from mmprep.common.logging_utils import stage_log
from setup_data import _content_range_total, _download_uci_dataset, _extract_zip_recursive, _iter_eegmmidb_targets, perform_setup


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_setup_creates_brief_dirs_manifest_and_progress(tmp_path, capsys):
    cfg = {
        "runtime": {"setup_mode": "strict", "http_timeout_seconds": 10, "retry_count": 1, "ptbxl_download_workers": 2},
        "datasets": {
            "pamap2": {"source_url": "https://example.com/pamap2.zip", "dataset_version": None, "required": True},
            "wisdm": {"source_url": "https://example.com/wisdm.zip", "dataset_version": None, "required": True},
            "mhealth": {"source_url": "https://example.com/mhealth.zip", "dataset_version": None, "required": False},
            "eegmmidb": {"source_url": "https://physionet.org/files/eegmmidb/1.0.0/", "dataset_version": "1.0.0", "required": True},
            "ptbxl": {"source_url": "https://physionet.org/files/ptb-xl/1.0.3/", "dataset_version": "1.0.3", "required": True},
        },
        "eeg": {"runs": [4, 8, 12], "subjects": [1, 2]},
        "ecg": {"sampling_rate_hz": 100, "max_records": 2},
    }
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "default.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    manifest = perform_setup(tmp_path, include_mhealth=False, simulate_downloads=True, config_path="configs/default.yaml")
    out = capsys.readouterr().out
    assert (tmp_path / "data/raw").exists()
    assert (tmp_path / "data/interim").exists()
    assert (tmp_path / "data/processed").exists()
    assert (tmp_path / "reports/download_manifest.json").exists()
    assert (tmp_path / "reports/setup_summary.md").exists()
    assert not any(x["dataset_name"] == "mhealth" for x in manifest)
    assert (tmp_path / "reports/compliance_checklist.json").exists()
    assert "Creating repository folders" in out
    assert "eegmmidb" in out


def test_config_carries_real_eeg_and_ecg_urls():
    cfg = yaml.safe_load((REPO_ROOT / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
    assert cfg['datasets']['eegmmidb']['source_url'].startswith('https://physionet.org/files/eegmmidb/')
    assert cfg['datasets']['ptbxl']['source_url'].startswith('https://physionet.org/files/ptb-xl/')


def test_eeg_target_iterator_matches_runs_and_subjects():
    targets = list(_iter_eegmmidb_targets(runs=[4, 12], subjects=[1, 9]))
    assert targets == [
        (1, 4, "S001/S001R04.edf"),
        (1, 12, "S001/S001R12.edf"),
        (9, 4, "S009/S009R04.edf"),
        (9, 12, "S009/S009R12.edf"),
    ]


def test_content_range_total_parses_complete_size():
    assert _content_range_total({"Content-Range": "bytes 0-0/123456"}) == 123456
    assert _content_range_total({"Content-Range": "bytes 100-200/*"}) is None
    assert _content_range_total({}) is None


def test_fresh_setup_resets_all_stage_metrics(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for stage in ["setup", "preprocess", "validate"]:
        (reports_dir / f"{stage}_metrics.json").write_text(
            json.dumps({"attempt_count": 7, "cumulative_duration_seconds": 10.0}),
            encoding="utf-8",
        )
    with stage_log("setup", tmp_path):
        pass
    setup_metrics = json.loads((reports_dir / "setup_metrics.json").read_text(encoding="utf-8"))
    assert setup_metrics["attempt_count"] == 1
    assert not (reports_dir / "preprocess_metrics.json").exists()
    assert not (reports_dir / "validate_metrics.json").exists()


def test_download_uci_dataset_redownloads_invalid_existing_archive(tmp_path, monkeypatch):
    out_dir = tmp_path / "data" / "raw" / "pamap2"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / "pamap2.zip"
    archive_path.write_bytes(b"PK\x03\x04truncated")

    calls: list[int] = []

    def fake_stream_http(url, destination, timeout, retries, show_progress=True):
        calls.append(1)
        with zipfile.ZipFile(destination, "w") as zf:
            zf.writestr("payload.txt", "ok")
        return {
            "status": "downloaded",
            "bytes": destination.stat().st_size,
            "remote_bytes": destination.stat().st_size,
            "fingerprint": "deadbeefcafebabe",
        }

    monkeypatch.setattr("setup_data._stream_http", fake_stream_http)

    record = _download_uci_dataset(
        name="pamap2",
        url="https://example.com/pamap2.zip",
        out_dir=out_dir,
        timeout=10,
        retries=2,
        remove_archives_after_extract=False,
        dataset_version=None,
    )

    assert len(calls) == 1
    assert record.status == "downloaded"
    assert (out_dir / "payload.txt").exists()
    assert json.loads((out_dir / "EXTRACTION_COMPLETE.json").read_text(encoding="utf-8"))["file_count"] >= 1


def test_extract_zip_recursive_removes_nested_archives_at_all_levels(tmp_path):
    out_dir = tmp_path / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    level2 = tmp_path / "level2.zip"
    with zipfile.ZipFile(level2, "w") as zf:
        zf.writestr("deep/file.txt", "ok")
    level1 = tmp_path / "level1.zip"
    with zipfile.ZipFile(level1, "w") as zf:
        zf.write(level2, arcname="inner/level2.zip")
    root_zip = out_dir / "root.zip"
    with zipfile.ZipFile(root_zip, "w") as zf:
        zf.write(level1, arcname="outer/level1.zip")

    extracted = _extract_zip_recursive(root_zip, out_dir, remove_archives_after_extract=True)

    assert extracted >= 3
    assert not list(out_dir.rglob("*.zip"))
    assert (out_dir / "outer" / "level1" / "inner" / "level2" / "deep" / "file.txt").exists()
