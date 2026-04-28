"""Validation helpers for processed outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mmprep.common.io_utils import load_yaml

REQUIRED_COLUMNS = {
    "sample_id", "dataset_name", "modality", "subject_or_patient_id", "source_file_or_record",
    "split", "label_or_event", "sampling_rate_hz", "n_channels", "n_samples", "channel_schema", "qc_flags",
}


def _status(ok: bool) -> str:
    """Return a compact markdown status marker."""
    return "PASS" if ok else "FAIL"


def summarize_npz(path: Path) -> dict:
    """Summarize one processed array bundle for integrity checks."""
    data = np.load(path, allow_pickle=False)
    x = data["X"]
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "has_nan": bool(np.isnan(x).any()),
        "has_inf": bool(np.isinf(x).any()),
    }


def _resolve_validation_config(manifest_path: Path, config_path: str | Path | None) -> dict | None:
    """Load the active config explicitly or infer it from reproducibility context."""
    candidates: list[Path] = []
    if config_path is not None:
        candidates.append(Path(config_path))
    reports_dir = manifest_path.parent
    context_path = reports_dir / "reproducibility_context.json"
    if context_path.exists():
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
            stored = context.get("config_path")
            if stored:
                candidates.append(Path(stored))
        except Exception:
            pass
    for candidate in candidates:
        if candidate.exists():
            return load_yaml(candidate)
    return None


def _check_har(df: pd.DataFrame, stem: str, lines: list[str], cfg: dict | None = None) -> bool:
    """Validate HAR-specific schema, windowing, and provenance expectations."""
    ok = True
    har_cfg = cfg.get("har", {}) if cfg else {}
    expected_channels = len(har_cfg.get("channel_schema", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]))
    expected_rate = int(har_cfg.get("target_hz", 20))
    expected_schema = "|".join(har_cfg.get("channel_schema", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]))
    pretrain_seconds = float(har_cfg.get("pretrain_window_seconds", 10))
    supervised_seconds = float(har_cfg.get("supervised_window_seconds", 5))
    pretrain_overlap = float(har_cfg.get("pretrain_overlap_seconds", 0))
    supervised_overlap = float(har_cfg.get("supervised_overlap_seconds", 2.5))
    if not (df["n_channels"] == expected_channels).all():
        lines.append("  - HAR channel count check failed")
        ok = False
    if not (df["sampling_rate_hz"] == expected_rate).all():
        lines.append("  - HAR sampling rate check failed")
        ok = False
    if not (df["channel_schema"].astype(str) == expected_schema).all():
        lines.append("  - HAR channel schema check failed")
        ok = False
    pretrain_samples = int(round(expected_rate * pretrain_seconds))
    supervised_samples = int(round(expected_rate * supervised_seconds))
    if "pretrain" in stem and not (df["n_samples"] == pretrain_samples).all():
        lines.append("  - HAR pretrain window length check failed")
        ok = False
    if "supervised" in stem and not (df["n_samples"] == supervised_samples).all():
        lines.append("  - HAR supervised window length check failed")
        ok = False
    if "supervised" in stem and "null_label_policy" not in df.columns:
        lines.append("  - HAR null-label policy field missing")
        ok = False
    if "supervised" in stem and "label_schema_name" not in df.columns:
        lines.append("  - HAR unified label schema field missing")
        ok = False
    if "supervised" in stem and "original_label" not in df.columns:
        lines.append("  - HAR original-label provenance field missing")
        ok = False
    required_window_cols = {"window_start_sample", "window_end_sample", "window_seconds"} - set(df.columns)
    if required_window_cols:
        lines.append(f"  - HAR window metadata missing fields: {sorted(required_window_cols)}")
        ok = False
    elif not df.empty:
        expected_samples = pretrain_samples if "pretrain" in stem else supervised_samples
        step_seconds = pretrain_seconds - pretrain_overlap if "pretrain" in stem else supervised_seconds - supervised_overlap
        expected_step = int(round(expected_rate * step_seconds))
        actual_lengths = (df["window_end_sample"] - df["window_start_sample"]).astype(int)
        if not (actual_lengths == expected_samples).all():
            lines.append("  - HAR window boundary check failed: end-start does not match expected window length")
            ok = False
        if not (df["window_start_sample"].astype(int) % expected_step == 0).all():
            lines.append("  - HAR window start alignment check failed")
            ok = False

        ordered = df.sort_values(["subject_or_patient_id", "source_file_or_record", "window_start_sample"]).reset_index(drop=True)
        diffs = (
            ordered.groupby(["subject_or_patient_id", "source_file_or_record"])["window_start_sample"]
            .diff()
            .dropna()
            .astype(int)
        )
        if not diffs.empty:
            non_positive = diffs <= 0
            misaligned = diffs % expected_step != 0
            if non_positive.any():
                lines.append("  - HAR window ordering check failed: window starts must increase within each source record")
                ok = False
            if misaligned.any():
                if "pretrain" in stem:
                    lines.append("  - HAR pretrain stride check failed: retained windows should advance in 10 s increments")
                else:
                    lines.append("  - HAR supervised stride check failed: retained windows should advance in 2.5 s increments")
                ok = False
    if df["sample_id"].duplicated().any():
        lines.append("  - HAR sample_id uniqueness failed")
        ok = False
    required_provenance = ["subject_or_patient_id", "source_file_or_record"]
    for col in required_provenance:
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            lines.append(f"  - HAR provenance field has missing values: {col}")
            ok = False
    if "pretrain" in stem:
        pretrain_labels = df["label_or_event"]
        if not (pretrain_labels.isna() | (pretrain_labels.astype(str).str.strip() == "")).all():
            lines.append("  - HAR pretrain outputs should be unlabeled")
            ok = False
    if "supervised" in stem:
        disallowed = df["label_or_event"].astype(str).str.strip().str.lower().isin({"0", "null", "transient", ""})
        if disallowed.any():
            lines.append("  - HAR supervised outputs contain null/transient labels")
            ok = False
    return ok


def _check_eeg(df: pd.DataFrame, lines: list[str], cfg: dict | None = None) -> bool:
    """Validate EEG event-window metadata and basic label constraints."""
    ok = True
    eeg_cfg = cfg.get("eeg", {}) if cfg else {}
    expected_rate = int(eeg_cfg.get("sampling_rate_hz", 160))
    expected_window_seconds = int(eeg_cfg.get("window_seconds", 4))
    expected_samples = expected_rate * expected_window_seconds
    allowed_runs = {str(int(run)) for run in eeg_cfg.get("runs", [4, 8, 12])}
    allowed_events = [str(code) for code in eeg_cfg.get("event_codes", ["T1", "T2"])]
    if bool(eeg_cfg.get("include_t0", False)) and "T0" not in allowed_events:
        allowed_events = ["T0", *allowed_events]
    allowed_events_set = set(allowed_events)
    missing = {"run_id", "event_onset", "event_onset_seconds", "event_code"} - set(df.columns)
    if missing:
        lines.append(f"  - EEG event metadata fields missing: {sorted(missing)}")
        ok = False
    if not (df["n_samples"] == expected_samples).all():
        lines.append("  - EEG 4-second window check failed")
        ok = False
    if not (df["sampling_rate_hz"] == expected_rate).all():
        lines.append("  - EEG sampling rate does not match the configured output rate")
        ok = False
    if "event_onset_seconds" in df.columns and (df["event_onset_seconds"] < 0).any():
        lines.append("  - EEG event onset seconds must be non-negative")
        ok = False
    if "label_or_event" in df.columns and not df["label_or_event"].astype(str).isin(sorted(allowed_events_set)).all():
        lines.append(f"  - EEG event labels contain codes outside the configured set: {sorted(allowed_events_set)}")
        ok = False
    if "run_id" in df.columns and not df["run_id"].astype(str).isin(sorted(allowed_runs)).all():
        lines.append(f"  - EEG run_id contains values outside the configured set: {sorted(allowed_runs)}")
        ok = False
    if {"event_code", "label_or_event"}.issubset(df.columns):
        if not (df["event_code"].astype(str) == df["label_or_event"].astype(str)).all():
            lines.append("  - EEG event_code does not match label_or_event")
            ok = False
    return ok


def _check_ptbxl(df: pd.DataFrame, lines: list[str], cfg: dict | None = None) -> bool:
    """Validate PTB-XL fold metadata and leakage-safe split fields."""
    ok = True
    ecg_cfg = cfg.get("ecg", {}) if cfg else {}
    holdout_fold = int(ecg_cfg.get("holdout_fold", 10))
    train_folds = {int(fold) for fold in ecg_cfg.get("train_folds", [1, 2, 3, 4, 5, 6, 7, 8, 9])}
    missing = {"strat_fold", "cv_fold", "lead_names"} - set(df.columns)
    if missing:
        lines.append(f"  - PTB-XL fold metadata missing: {sorted(missing)}")
        ok = False
    if "patient_safe_split" in df.columns and not df["patient_safe_split"].astype(bool).all():
        lines.append("  - PTB-XL patient_safe_split should be true for all rows")
        ok = False
    if "split" in df.columns and not df["split"].isin(["train", "test"]).all():
        lines.append("  - PTB-XL split must be train/test")
        ok = False
    if {"split", "cv_fold", "strat_fold"}.issubset(df.columns):
        strat_folds = pd.to_numeric(df["strat_fold"], errors="coerce")
        cv_folds = pd.to_numeric(df["cv_fold"], errors="coerce")
        test_rows = df["split"] == "test"
        train_rows = df["split"] == "train"
        bad = df[test_rows & df["cv_fold"].notna()]
        if not bad.empty:
            lines.append("  - PTB-XL holdout leakage check failed: test rows should not have cv_fold")
            ok = False
        bad_holdout = df[test_rows & (strat_folds != holdout_fold)]
        if not bad_holdout.empty:
            lines.append(f"  - PTB-XL holdout fold check failed: test rows must use configured holdout_fold={holdout_fold}")
            ok = False
        bad_train = df[train_rows & ~strat_folds.isin(sorted(train_folds))]
        if not bad_train.empty:
            lines.append(f"  - PTB-XL train fold check failed: train rows must use configured train_folds={sorted(train_folds)}")
            ok = False
        bad_cv = df[train_rows & (cv_folds != strat_folds)]
        if not bad_cv.empty:
            lines.append("  - PTB-XL cv_fold check failed: train rows should retain their strat_fold as cv_fold")
            ok = False
    if {"split", "subject_or_patient_id"}.issubset(df.columns):
        train_patients = set(df.loc[df["split"] == "train", "subject_or_patient_id"].astype(str))
        test_patients = set(df.loc[df["split"] == "test", "subject_or_patient_id"].astype(str))
        overlap = train_patients & test_patients
        if overlap:
            lines.append(f"  - PTB-XL patient leakage detected across train/test splits: {sorted(overlap)[:5]}")
            ok = False
        if "patient_safe_split" in df.columns:
            expected = not overlap
            observed = df["patient_safe_split"].fillna(False).astype(bool)
            if not (observed == expected).all():
                lines.append("  - PTB-XL patient_safe_split flag does not match the actual split partition")
                ok = False
    if {"lead_names", "channel_schema"}.issubset(df.columns):
        if not (df["lead_names"].astype(str) == df["channel_schema"].astype(str)).all():
            lines.append("  - PTB-XL lead_names does not match channel_schema")
            ok = False
    return ok


def _reproducibility_evidence(processed_dir: Path, manifest_path: Path) -> tuple[bool, str]:
    """Summarize reproducibility signals available from the current tree."""
    if not processed_dir.exists():
        return False, f"processed directory missing: {processed_dir}"

    data_dir = processed_dir.parent
    root_like = data_dir.parent if data_dir.name == "data" else data_dir
    expected_dirs = []
    if data_dir.name == "data":
        expected_dirs = [data_dir / "raw", data_dir / "interim", data_dir / "processed", root_like / "reports"]

    missing_dirs = [str(path) for path in expected_dirs if not path.exists()]
    if not manifest_path.exists():
        return False, f"processed manifest missing: {manifest_path}"

    required_docs = [
        root_like / "README.md",
        root_like / "preprocessing_plan.md",
        root_like / "reports" / "download_manifest.json",
        root_like / "reports" / "setup_summary.md",
        root_like / "reports" / "resource_estimate.md",
        root_like / "reports" / "reproducibility_context.json",
    ]
    missing_docs = [str(path) for path in required_docs if not path.exists()]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        return False, "processed manifest is empty"

    download_manifest = root_like / "reports" / "download_manifest.json"
    required_dataset_ok = True
    dataset_evidence = "download_manifest_unavailable"
    if download_manifest.exists():
        try:
            download_rows = json.loads(download_manifest.read_text(encoding="utf-8"))
            status_map = {str(row.get("dataset_name")): str(row.get("status")) for row in download_rows}
            required_names = {"pamap2", "wisdm", "eegmmidb", "ptbxl"}
            required_dataset_ok = required_names.issubset(status_map) and all(status_map[name] != "failed" for name in required_names if name in status_map)
            dataset_evidence = f"required_dataset_statuses={{{', '.join(f'{name}:{status_map.get(name, 'missing')}' for name in sorted(required_names))}}}"
        except Exception:
            required_dataset_ok = False
            dataset_evidence = "download_manifest_parse_failed"

    context_ok = True
    context_evidence = "reproducibility_context_unavailable"
    context_path = root_like / "reports" / "reproducibility_context.json"
    if context_path.exists():
        try:
            context = json.loads(context_path.read_text(encoding="utf-8"))
            required_keys = {"config_path", "config_sha256", "python_version", "platform"}
            missing_context = sorted(required_keys - set(context))
            config_sha = context.get("config_sha256")
            config_path = Path(context.get("config_path", ""))
            computed_sha = None
            if config_path.exists():
                computed_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            completed_stages = set(str(stage) for stage in context.get("completed_stages", []))
            missing_stages = sorted({"setup", "preprocess", "validate"} - completed_stages)
            context_ok = (
                not missing_context
                and bool(config_sha)
                and (computed_sha is None or computed_sha == config_sha)
                and not missing_stages
            )
            context_evidence = (
                f"context_keys_missing={len(missing_context)}, "
                f"completed_stages={sorted(completed_stages)}, missing_stages={missing_stages}"
            )
        except Exception:
            context_ok = False
            context_evidence = "reproducibility_context_parse_failed"

    manifest_paths = {str(Path(entry["path"]).resolve()) for entry in manifest if entry.get("path")}
    processed_files = {
        str(path.resolve())
        for path in processed_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in {".npz", ".csv"}
    }
    missing_from_manifest = sorted(processed_files - manifest_paths)
    missing_on_disk = sorted(path for path in manifest_paths if not Path(path).exists())
    latest_processed_mtime = max((path.stat().st_mtime for path in processed_dir.glob("**/*") if path.is_file()), default=0.0)
    manifest_fresh = manifest_path.stat().st_mtime >= latest_processed_mtime if manifest_path.exists() else False

    ok = (
        not missing_dirs
        and not missing_docs
        and required_dataset_ok
        and context_ok
        and not missing_from_manifest
        and not missing_on_disk
        and manifest_fresh
        and bool(processed_files)
    )
    evidence_parts = []
    if expected_dirs:
        evidence_parts.append(f"stage_dirs_present={len(expected_dirs) - len(missing_dirs)}/{len(expected_dirs)}")
    evidence_parts.append(f"processed_files={len(processed_files)}")
    evidence_parts.append(f"manifest_entries={len(manifest)}")
    evidence_parts.append(f"manifest_fresh={manifest_fresh}")
    evidence_parts.append(dataset_evidence)
    evidence_parts.append(context_evidence)
    if missing_dirs:
        evidence_parts.append(f"missing_dirs={len(missing_dirs)}")
    if missing_docs:
        evidence_parts.append(f"missing_docs={len(missing_docs)}")
    if missing_from_manifest:
        evidence_parts.append(f"unmanifested_files={len(missing_from_manifest)}")
    if missing_on_disk:
        evidence_parts.append(f"missing_manifest_targets={len(missing_on_disk)}")
    return ok, ", ".join(evidence_parts)


def validate_processed(
    processed_dir: str | Path,
    manifest_path: str | Path,
    sample_dir: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[bool, str]:
    """Validate processed outputs and render the markdown validation report.

    Inputs:
    - processed_dir: directory containing processed `.npz` and `.csv` bundles.
    - manifest_path: expected processed manifest path.
    - sample_dir: optional representative sample-pack directory.

    Outputs:
    - Tuple of `(ok, report_markdown)`.
    """
    processed_dir = Path(processed_dir)
    manifest_path = Path(manifest_path)
    sample_dir = Path(sample_dir) if sample_dir else None
    cfg = _resolve_validation_config(manifest_path, config_path)
    lines = ["# Validation report"]
    ok = True
    har_supervised_schemas: dict[str, str] = {}
    har_all_schemas: dict[str, set[str]] = {}
    har_counts: list[str] = []
    eeg_counts: list[str] = []
    ecg_counts: list[str] = []
    sample_csv_count = 0
    manifest_ok = False
    reproducibility_ok = False
    reproducibility_evidence = ""
    total_npz = 0
    arrays_ok = True
    har_ok_all = True
    eeg_ok_all = True
    ecg_ok_all = True
    har_window_ok = True
    har_null_ok = True
    eeg_evidence: pd.DataFrame | None = None
    ecg_evidence: pd.DataFrame | None = None
    sample_checks: list[str] = []
    sample_pack_ok = True

    lines += ["", "## Required Validation Checks", "", "| Check | Status | Evidence |", "|---|---|---|"]
    detail_lines = ["", "## Array integrity and schema checks", ""]

    for npz in sorted(processed_dir.glob("**/*.npz")):
        total_npz += 1
        info = summarize_npz(npz)
        detail_lines.append(f"- **{npz.relative_to(processed_dir)}**: shape={info['shape']}, dtype={info['dtype']}, has_nan={info['has_nan']}, has_inf={info['has_inf']}")
        if info["dtype"] != "float32":
            detail_lines.append("  - array dtype should be float32")
            ok = False
            arrays_ok = False
        if info["has_nan"] or info["has_inf"]:
            ok = False
            arrays_ok = False
        csv_path = npz.with_suffix('.csv')
        if not csv_path.exists():
            detail_lines.append("  - metadata CSV missing")
            ok = False
            arrays_ok = False
            continue
        df = pd.read_csv(csv_path)
        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        detail_lines.append(f"  - metadata rows={len(df)}, missing_required_columns={missing}")
        if missing:
            ok = False
            arrays_ok = False
        if len(df) != info['shape'][0]:
            detail_lines.append("  - row count does not match array batch dimension")
            ok = False
            arrays_ok = False
        if df["sample_id"].duplicated().any():
            detail_lines.append("  - duplicate sample_id values detected")
            ok = False
            arrays_ok = False
        if any(tag in npz.stem for tag in ["pamap2", "wisdm", "mhealth"]):
            har_ok = _check_har(df, npz.stem, detail_lines, cfg=cfg)
            ok = har_ok and ok
            arrays_ok = har_ok and arrays_ok
            har_ok_all = har_ok and har_ok_all
            har_window_ok = har_ok and har_window_ok
            har_null_ok = har_ok and har_null_ok
            if "supervised" in npz.stem and "label_schema_name" in df.columns and not df.empty:
                har_supervised_schemas[npz.stem] = str(df["label_schema_name"].iloc[0])
            if not df.empty:
                har_all_schemas[npz.stem] = set(df["channel_schema"].astype(str).unique().tolist())
                har_counts.append(f"- {npz.stem}: windows={len(df)}, subjects={df['subject_or_patient_id'].nunique()}, labels={sorted(df['label_or_event'].dropna().astype(str).unique().tolist())[:10]}")
        if "eegmmidb" in npz.stem:
            eeg_ok = _check_eeg(df, detail_lines, cfg=cfg)
            ok = eeg_ok and ok
            arrays_ok = eeg_ok and arrays_ok
            eeg_ok_all = eeg_ok and eeg_ok_all
            if not df.empty:
                eeg_evidence = df.copy()
                eeg_counts.append(
                    f"- {npz.stem}: windows={len(df)}, subjects={df['subject_or_patient_id'].nunique()}, runs={df['run_id'].nunique()}, events={df['label_or_event'].value_counts().to_dict()}"
                )
        if "ptbxl" in npz.stem:
            ecg_ok = _check_ptbxl(df, detail_lines, cfg=cfg)
            ok = ecg_ok and ok
            arrays_ok = ecg_ok and arrays_ok
            ecg_ok_all = ecg_ok and ecg_ok_all
            if not df.empty:
                ecg_evidence = df.copy()
                split_counts = df["split"].value_counts(dropna=False).to_dict()
                fold_counts = df["strat_fold"].value_counts(dropna=False).sort_index().to_dict() if "strat_fold" in df.columns else {}
                ecg_counts.append(
                    f"- {npz.stem}: records={len(df)}, patients={df['subject_or_patient_id'].nunique()}, splits={split_counts}, strat_folds={fold_counts}"
                )

    if total_npz == 0:
        detail_lines.append("- no processed array bundles were found under the processed directory")
        ok = False
        arrays_ok = False
        har_ok_all = False
        eeg_ok_all = False
        ecg_ok_all = False
        har_window_ok = False
        har_null_ok = False

    har_schema_ok = True
    har_channel_schema_ok = True
    har_label_space_ok = True
    expected_har_label_schema = (
        str(cfg.get("har", {}).get("label_schema_name", "har_unified_v1")).strip()
        if cfg
        else "har_unified_v1"
    )
    if har_supervised_schemas:
        distinct = sorted(set(har_supervised_schemas.values()))
        har_schema_ok = len(distinct) == 1 and distinct[0] == expected_har_label_schema
        lines += ["| HAR harmonisation: datasets share the configured schema name | "
                  f"{_status(har_schema_ok)} | detected label schema names: {distinct}; expected: {expected_har_label_schema} |"]
        if not har_schema_ok:
            ok = False
    else:
        har_schema_ok = False
        lines += ["| HAR harmonisation: datasets share the configured schema name | FAIL | no supervised HAR schema evidence found |"]

    har_schema_strings = sorted({schema for values in har_all_schemas.values() for schema in values})
    expected_har_schema = "|".join(cfg.get("har", {}).get("channel_schema", ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])) if cfg else "acc_x|acc_y|acc_z|gyro_x|gyro_y|gyro_z"
    har_channel_schema_ok = har_schema_strings == [expected_har_schema] if har_schema_strings else False
    if not har_channel_schema_ok:
        ok = False

    allowed_har_labels = (
        {str(v).strip() for mapping in cfg.get("har", {}).get("label_maps", {}).values() for v in mapping.values()}
        if cfg and cfg.get("har", {}).get("label_maps")
        else {"lying", "sitting", "standing", "walking", "running", "cycling", "stairs", "household", "other"}
    )
    supervised_har_labels = set()
    for desc in har_counts:
        pass
    for npz in sorted(processed_dir.glob("**/*.npz")):
        if any(tag in npz.stem for tag in ["pamap2", "wisdm", "mhealth"]) and "supervised" in npz.stem:
            csv_path = npz.with_suffix(".csv")
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                supervised_har_labels.update(
                    value
                    for value in df["label_or_event"].dropna().astype(str).str.strip().tolist()
                    if value
                )
    unexpected_har_labels = sorted(supervised_har_labels - allowed_har_labels)
    har_label_space_ok = not unexpected_har_labels
    if not har_label_space_ok:
        ok = False

    lines += [
        f"| Array integrity: float32, no NaN/Inf, row counts, required metadata | {_status(arrays_ok)} | checked {total_npz} processed array bundles |",
        f"| HAR harmonisation: identical shared channel schema across datasets | {_status(har_channel_schema_ok)} | detected schemas: {har_schema_strings if har_schema_strings else 'none'} |",
        f"| HAR harmonisation: supervised labels stay within the unified label set | {_status(har_label_space_ok)} | unexpected labels: {unexpected_har_labels if unexpected_har_labels else 'none'} |",
        f"| HAR window contract: 10 s pretrain, 5 s supervised, configured stride alignment | {_status(har_window_ok)} | per-file HAR checks include length, stride alignment, and monotonic starts |",
        f"| HAR null/transient handling documented and fields present | {_status(har_null_ok)} | null-label policy and original-label provenance checked in supervised HAR metadata |",
        f"| EEG annotation correctness | {_status(eeg_ok_all and bool(eeg_counts))} | see EEG evidence section for configured event counts and run coverage |",
        f"| ECG fold correctness and leakage control | {_status(ecg_ok_all and bool(ecg_counts))} | see ECG evidence section for split and stratified fold counts |",
    ]

    if har_counts:
        lines += [
            "",
            "## HAR harmonization cross-check",
            "",
            f"- detected label schema names: {sorted(set(har_supervised_schemas.values())) if har_supervised_schemas else []}",
            f"- expected label schema name: {expected_har_label_schema}",
            f"- detected shared channel schemas: {har_schema_strings if har_schema_strings else []}",
            f"- allowed supervised label set: {sorted(allowed_har_labels)}",
            f"- unexpected supervised labels: {unexpected_har_labels if unexpected_har_labels else []}",
            "",
            "## HAR evidence",
            "",
        ]
        lines.extend(har_counts)
    if eeg_counts:
        lines += ["", "## EEG evidence", ""]
        lines.extend(eeg_counts)
        if eeg_evidence is not None:
            run_event = (
                eeg_evidence.groupby(["run_id", "label_or_event"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=["T1", "T2"], fill_value=0)
                .sort_index()
            )
            first_t1 = (
                eeg_evidence[eeg_evidence["label_or_event"] == "T1"]
                .groupby("run_id")["event_onset_seconds"]
                .min()
            )
            lines += ["", "### EEG Annotation Table", "", "| Run | T1 events | T2 events | First T1 onset (s) |", "|---|---:|---:|---:|"]
            for run_id, row in run_event.iterrows():
                onset = first_t1.get(run_id, np.nan)
                onset_txt = "NA" if pd.isna(onset) else f"{float(onset):.3f}"
                lines.append(f"| {run_id} | {int(row.get('T1', 0))} | {int(row.get('T2', 0))} | {onset_txt} |")
    if ecg_counts:
        lines += ["", "## ECG evidence", ""]
        lines.extend(ecg_counts)
        if ecg_evidence is not None:
            split_counts = ecg_evidence["split"].value_counts().to_dict()
            total = max(len(ecg_evidence), 1)
            train_only = ecg_evidence[ecg_evidence["split"] == "train"]
            cv_counts = train_only["cv_fold"].value_counts(dropna=False).sort_index().to_dict()
            train_patients = set(train_only["subject_or_patient_id"].astype(str).unique())
            test_patients = set(ecg_evidence[ecg_evidence["split"] == "test"]["subject_or_patient_id"].astype(str).unique())
            leakage = bool(train_patients & test_patients)
            lines += ["", "### ECG Fold Table", "", "| Split | Count | Percentage | strat_fold values |", "|---|---:|---:|---|"]
            holdout_fold = int(cfg.get("ecg", {}).get("holdout_fold", 10)) if cfg else 10
            train_folds = cfg.get("ecg", {}).get("train_folds", [1, 2, 3, 4, 5, 6, 7, 8, 9]) if cfg else [1, 2, 3, 4, 5, 6, 7, 8, 9]
            train_folds_txt = ",".join(str(int(fold)) for fold in train_folds)
            for split in ["train", "test"]:
                count = int(split_counts.get(split, 0))
                pct = 100.0 * count / total
                folds = train_folds_txt if split == "train" else str(holdout_fold)
                lines.append(f"| {split} | {count} | {pct:.2f}% | {folds} |")
            lines += ["", "**CV fold distribution (train only):**", "", "| cv_fold | Count |", "|---|---:|"]
            for fold, count in cv_counts.items():
                lines.append(f"| {int(fold)} | {int(count)} |")
            lines += ["", f"**Leakage check**: {'No' if not leakage else 'Yes'} patient appears in both train and test splits."]

    lines += ["", "## Manifest checks", ""]
    if not manifest_path.exists():
        lines.append("- processed manifest missing")
        ok = False
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        lines.append(f"- processed manifest entries: {len(manifest)}")
        if not manifest:
            ok = False
        else:
            manifest_ok = True
    reproducibility_ok, reproducibility_evidence = _reproducibility_evidence(processed_dir, manifest_path)
    ok = ok and reproducibility_ok

    raw_storage = sum(p.stat().st_size for p in processed_dir.parents[0].joinpath("raw").glob("**/*") if p.is_file()) if processed_dir.parents else 0
    interim_dir = processed_dir.parents[0].joinpath("interim") if processed_dir.parents else Path("data/interim")
    interim_storage = sum(p.stat().st_size for p in interim_dir.glob("**/*") if p.is_file()) if interim_dir.exists() else 0
    processed_storage = sum(p.stat().st_size for p in processed_dir.glob("**/*") if p.is_file()) if processed_dir.exists() else 0
    lines += [
        "",
        "## Resource awareness",
        "",
        f"- raw storage bytes: {raw_storage}",
        f"- interim storage bytes: {interim_storage}",
        f"- processed storage bytes: {processed_storage}",
        f"- total processed arrays: {len(list(processed_dir.glob('**/*.npz')))}",
        "- chunking strategy: HAR file-by-file, EEG EDF-by-EDF, ECG record-by-record.",
        "- full peak RAM and runtime details: see `reports/resource_estimate.md`.",
    ]

    if sample_dir:
        lines += ["", "## Representative sample checks", ""]
        if sample_dir.exists():
            files = sorted(sample_dir.glob("**/*.csv"))
            sample_csv_count = len(files)
            lines.append(f"- sample metadata files: {sample_csv_count}")
            if sample_csv_count == 0:
                lines.append("- representative sample pack has not been generated yet")
            manifest_sample_path = sample_dir / "sample_pack_manifest.json"
            if sample_csv_count == 0 and not manifest_sample_path.exists():
                sample_pack_ok = True
            elif not manifest_sample_path.exists():
                lines.append("- sample_pack_manifest.json missing")
                ok = False
                sample_pack_ok = False
            else:
                sample_manifest = json.loads(manifest_sample_path.read_text(encoding="utf-8"))
                dataset_expected_totals: dict[str, int] = {}
                dataset_actual_totals: dict[str, int] = {}
                for item in sample_manifest.get("files", []):
                    sample_csv = Path(item["sample_csv"])
                    sample_npz = Path(item["sample_npz"])
                    source_csv = Path(item["source_csv"])
                    source_npz = Path(item["source_npz"])
                    expected_rows = int(item["expected_rows"])
                    expected_shape = tuple(int(x) for x in item["expected_shape"])
                    dataset_name = str(item["dataset_name"])
                    dataset_expected_totals[dataset_name] = int(item["dataset_target_rows"])
                    if not sample_csv.exists() or not sample_npz.exists():
                        lines.append(f"  - missing sample artefact for {dataset_name}: {sample_csv.name}")
                        ok = False
                        sample_pack_ok = False
                        continue
                    if not source_csv.exists() or not source_npz.exists():
                        lines.append(f"  - missing source processed artefact for {dataset_name}: {source_csv.name}")
                        ok = False
                        sample_pack_ok = False
                        continue
                    df = pd.read_csv(sample_csv)
                    arr = np.load(sample_npz, allow_pickle=False)["X"]
                    source_df = pd.read_csv(source_csv)
                    source_arr = np.load(source_npz, allow_pickle=False)["X"]
                    source_rows = min(len(source_df), int(source_arr.shape[0]))
                    target_rows = min(int(sample_manifest.get("target_rows_per_dataset", 100)), source_rows)
                    if len(df) != expected_rows:
                        lines.append(f"  - {sample_csv.name} row count {len(df)} does not match expected {expected_rows}")
                        ok = False
                        sample_pack_ok = False
                    if arr.shape[0] != len(df):
                        lines.append(f"  - {sample_npz.name} batch dimension does not match {sample_csv.name}")
                        ok = False
                        sample_pack_ok = False
                    if tuple(arr.shape) != expected_shape:
                        lines.append(f"  - {sample_npz.name} shape {tuple(arr.shape)} does not match expected {expected_shape}")
                        ok = False
                        sample_pack_ok = False
                    if expected_rows != min(expected_rows, source_rows):
                        lines.append(f"  - {sample_csv.name} expected_rows exceeds source size")
                        ok = False
                        sample_pack_ok = False
                    dataset_actual_totals[dataset_name] = dataset_actual_totals.get(dataset_name, 0) + len(df)
                    sample_checks.append(
                        f"| {sample_npz.relative_to(sample_dir)} | {expected_shape} | {tuple(arr.shape)} | "
                        f"{_status(len(df) == expected_rows and tuple(arr.shape) == expected_shape)} |"
                    )
                for dataset_name, expected_total in sorted(dataset_expected_totals.items()):
                    actual_total = dataset_actual_totals.get(dataset_name, 0)
                    lines.append(f"- dataset {dataset_name}: sample rows={actual_total}/{expected_total}")
                    if actual_total != expected_total:
                        lines.append(f"  - dataset {dataset_name} has {actual_total} sample rows, expected {expected_total}")
                        ok = False
                        sample_pack_ok = False
        else:
            lines.append("- submission_sample directory missing or not generated yet")
            sample_pack_ok = True

    lines[6:6] = [
        f"| Reproducibility: stage directories present and processed files agree with the manifest | {_status(reproducibility_ok)} | {reproducibility_evidence} |",
        f"| Sample pack status | {_status(sample_pack_ok and sample_csv_count > 0)} | sample metadata files detected: {sample_csv_count} |",
        f"| Resource awareness: storage totals reported here and peak RAM/runtime linked | PASS | raw={raw_storage} B, interim={interim_storage} B, processed={processed_storage} B; see `reports/resource_estimate.md` |",
    ]

    lines += ["", "## Data integrity checks", ""]
    if har_counts:
        lines += [
            "### HAR",
            f"- {_status(har_ok_all)} no NaN/Inf, fixed channel schema, and required HAR metadata fields",
            f"- {_status(har_window_ok)} window lengths and stride alignment match the configured contract",
        ]
    if eeg_evidence is not None:
        eeg_monotonic = bool(
            eeg_evidence.sort_values(["subject_or_patient_id", "run_id", "event_onset"])
            .groupby(["subject_or_patient_id", "run_id"])["event_onset"]
            .diff()
            .dropna()
            .ge(0)
            .all()
        )
        eeg_channels_ok = bool((eeg_evidence["n_channels"] == 64).all())
        lines += [
            "### EEG",
            f"- {_status(eeg_ok_all)} all retained EEG event windows passed shape and label checks",
            f"- {_status(eeg_channels_ok)} channel count is 64 across all retained windows",
            f"- {_status(eeg_monotonic)} event onsets are monotonic within each subject/run",
        ]
    if ecg_evidence is not None:
        expected_leads = cfg.get("ecg", {}).get("lead_order", ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]) if cfg else ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        expected_rate = int(cfg.get("ecg", {}).get("sampling_rate_hz", int(ecg_evidence["sampling_rate_hz"].iloc[0]))) if cfg else int(ecg_evidence["sampling_rate_hz"].iloc[0])
        lead_ok = bool((ecg_evidence["channel_schema"].astype(str) == "|".join(expected_leads)).all())
        rate_ok = bool((ecg_evidence["sampling_rate_hz"] == expected_rate).all())
        lines += [
            "### ECG",
            f"- {_status(ecg_ok_all)} PTB-XL rows passed fold and split checks",
            f"- {_status(lead_ok)} lead names are consistent with the configured lead order",
            f"- {_status(rate_ok)} sampling rate matches the configured ECG output rate",
        ]

    if sample_checks:
        lines += ["", "## Sample pack verification", "", "| File | Expected Shape | Actual Shape | Status |", "|---|---|---|---|"]
        lines.extend(sample_checks)

    lines.extend(detail_lines)
    lines += ["", "## Overall status", "", ("PASS" if ok else "FAIL"), ""]
    return ok, "\n".join(lines)
