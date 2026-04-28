"""Main pipeline orchestration.

The pipeline is intentionally staged as:
raw -> interim -> processed -> reports

Resource-aware design choices:
- HAR source records are parsed one file at a time before bundle assembly.
- EEG EDF files are read one file at a time and windowed event-by-event.
- PTB-XL records are read one record at a time from metadata.
This avoids loading every raw modality into memory at once, although final bundle assembly can
still dominate memory for very large HAR outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmprep.common.io_utils import ensure_dir, load_yaml, write_array_npz, write_dataframe, write_json
from mmprep.common.console import status_text
from mmprep.common.manifests import file_manifest_entry
from mmprep.common.progress import ProgressPrinter
from mmprep.common.reproducibility import update_reproducibility_context
from mmprep.common.signal_utils import resample_array
from mmprep.har.base import DEFAULT_HAR_LABEL_SCHEMA_NAME, clean_and_resample_har, harmonize_har_labels, make_har_windows
from mmprep.har.pamap2 import parse_pamap2_to_interim
from mmprep.har.wisdm import parse_wisdm_to_interim
from mmprep.har.mhealth import parse_mhealth_to_interim
from mmprep.eeg.eegmmidb import parse_eeg_csv_to_interim, parse_eeg_edf_to_interim, event_windows, infer_subject_run_from_name
from mmprep.ecg.ptbxl import parse_ptbxl_csv_to_interim, load_ptbxl_metadata, load_ptbxl_record, record_to_processed

def _bundle_dataset(out_dir: Path, stem: str, array: np.ndarray, rows: list[dict]) -> list[dict]:
    """Write one processed array bundle and its metadata CSV.

    Inputs:
    - out_dir: destination directory for processed outputs.
    - stem: shared filename stem for the array and CSV.
    - array: processed tensor to store under key `X`.
    - rows: metadata rows aligned to the first array dimension.

    Outputs:
    - Manifest entries for the written files.
    """
    ensure_dir(out_dir)
    arr_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.csv"
    write_array_npz(arr_path, array)
    write_dataframe(meta_path, pd.DataFrame(rows))
    return [
        file_manifest_entry(arr_path, {"shape": list(array.shape), "n_rows": int(array.shape[0])}),
        file_manifest_entry(meta_path, {"n_rows": len(rows)}),
    ]


def _bundle_exists(out_dir: Path, stem: str) -> bool:
    """Return whether a processed array bundle already exists."""
    return (out_dir / f"{stem}.npz").exists() and (out_dir / f"{stem}.csv").exists()


def _bundle_manifest(out_dir: Path, stem: str) -> list[dict]:
    """Return manifest entries for an already-existing processed bundle."""
    arr_path = out_dir / f"{stem}.npz"
    meta_path = out_dir / f"{stem}.csv"
    shape = list(np.load(arr_path, allow_pickle=False)["X"].shape)
    n_rows = len(pd.read_csv(meta_path))
    return [
        file_manifest_entry(arr_path, {"shape": shape, "n_rows": int(shape[0])}),
        file_manifest_entry(meta_path, {"n_rows": n_rows}),
    ]


def _write_interim_csv(path: Path, df: pd.DataFrame) -> None:
    """Write an interim CSV table, creating parent folders as needed."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def _write_interim_signal_npz(path: Path, array: np.ndarray) -> None:
    """Write an interim compressed signal array under key `X`."""
    ensure_dir(path.parent)
    np.savez_compressed(path, X=array.astype(np.float32, copy=False))


def _write_interim_summary(path: Path, payload: dict) -> None:
    """Write a lightweight interim JSON summary instead of duplicating raw signals."""
    ensure_dir(path.parent)
    write_json(path, payload)


def _discover_har_files(dataset_name: str, dataset_raw: Path) -> list[Path]:
    """Discover real HAR source files while excluding setup markers and docs."""
    if dataset_name == "pamap2":
        files = sorted(dataset_raw.rglob("subject*.dat")) + sorted(dataset_raw.rglob("*.csv"))
        return sorted({p.resolve(): p for p in files}.values(), key=lambda p: str(p))
    if dataset_name == "wisdm":
        watch_accel = sorted(dataset_raw.rglob("data_*_accel_watch.txt"))
        csvs = sorted(dataset_raw.rglob("*.csv"))
        return sorted({p.resolve(): p for p in [*watch_accel, *csvs]}.values(), key=lambda p: str(p))
    if dataset_name == "mhealth":
        logs = sorted(dataset_raw.rglob("mHealth_subject*.log"))
        csvs = sorted(dataset_raw.rglob("*.csv"))
        deduped: dict[str, Path] = {}
        for path in [*logs, *csvs]:
            key = path.name.lower()
            current = deduped.get(key)
            # Prefer the shallower path when archives unpack duplicate wrapper folders.
            if current is None or len(path.parts) < len(current.parts):
                deduped[key] = path
        return sorted(deduped.values(), key=lambda p: str(p))
    return []


def _har_source_id(dataset_raw: Path, file: Path) -> str:
    """Return a stable relative identifier for HAR provenance fields."""
    try:
        return file.relative_to(dataset_raw).as_posix()
    except Exception:
        return file.name


def _iter_har_subject_frames(df: pd.DataFrame, default_subject_id: str) -> list[tuple[str, pd.DataFrame]]:
    """Split a HAR frame into subject-specific subframes when subject IDs are present."""
    subject_col = next((col for col in ["subject_id", "subject_or_patient_id"] if col in df.columns), None)
    if subject_col is None:
        return [(default_subject_id, df.copy())]
    groups: list[tuple[str, pd.DataFrame]] = []
    for subject_value, subject_df in df.groupby(subject_col, sort=False):
        payload = subject_df.drop(columns=[subject_col]).reset_index(drop=True)
        groups.append((str(subject_value), payload))
    return groups or [(default_subject_id, df.copy())]




def _empty_har_array(kind: str, n_channels: int, sampling_rate_hz: int) -> np.ndarray:
    """Return an empty HAR bundle with the correct final window shape."""
    window_seconds = 10 if kind == "pretrain" else 5
    return np.empty((0, n_channels, sampling_rate_hz * window_seconds), dtype=np.float32)


def _manifest_dataset_bucket(rel_path: str) -> str:
    """Infer a dataset-level bucket name from one processed manifest path."""
    rel = Path(rel_path)
    stem = rel.stem
    if not stem:
        return "unknown"
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def _write_compliance_outputs(reports_dir: Path, manifest: list[dict], issues: list[str], cfg: dict) -> None:
    """Write a machine-readable and human-readable compliance summary after preprocessing."""
    checklist = {
        "preprocess_mode": cfg.get("runtime", {}).get("preprocess_mode", "best-effort"),
        "interim_stage_materialized": True,
        "processed_manifest_written": True,
        "dataset_outputs": {},
        "issues": issues,
    }
    for entry in manifest:
        rel = entry.get("path", "")
        dataset_bucket = _manifest_dataset_bucket(rel)
        checklist["dataset_outputs"].setdefault(dataset_bucket, 0)
        checklist["dataset_outputs"][dataset_bucket] += 1
    write_json(reports_dir / "compliance_checklist.json", checklist)


def _finalize_ptbxl_split_metadata(rows: list[dict]) -> None:
    """Mark whether the assembled PTB-XL split is actually patient-safe."""
    train_patients = {
        str(row["subject_or_patient_id"])
        for row in rows
        if str(row.get("split", "")).lower() == "train"
    }
    test_patients = {
        str(row["subject_or_patient_id"])
        for row in rows
        if str(row.get("split", "")).lower() == "test"
    }
    leakage = bool(train_patients & test_patients)
    for row in rows:
        row["patient_safe_split"] = not leakage

def _dataset_missing(name: str, required: bool, mode: str, issues: list[str]) -> None:
    """Record or raise a missing-dataset condition according to runtime mode."""
    msg = f"Dataset '{name}' is missing raw input files."
    if required and mode == "strict":
        raise RuntimeError(msg)
    issues.append(msg)
    print(f"  {status_text('WARNING', 'warn')} {msg} Skipping in {mode} mode.", flush=True)


def run_preprocess(config_path: str | Path, resume: bool = False) -> int:
    """Run the full raw-to-processed multimodal pipeline.

    Inputs:
    - config_path: YAML configuration path.
    - resume: whether to skip processed bundles that already exist.

    Outputs:
    - Zero on success after writing processed outputs and manifests.
    """
    cfg = load_yaml(config_path)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    ensure_dir(interim_dir)
    ensure_dir(processed_dir)
    ensure_dir(reports_dir)
    update_reproducibility_context(reports_dir, config_path, "preprocess")

    mode = cfg.get("runtime", {}).get("preprocess_mode", "best-effort")
    dataset_requirements = {k: bool(v.get("required", True)) for k, v in cfg.get("datasets", {}).items()}
    issues: list[str] = []
    har_label_maps = cfg.get("har", {}).get("label_maps", {})

    progress = ProgressPrinter(total_steps=5)
    progress.stage(f"Loading configuration and preparing output folders (mode={mode})")

    manifest = []
    schema = cfg["har"]["channel_schema"]
    null_labels = set([x for x in cfg["har"]["null_labels"] if x is not None])
    har_label_policy = str(cfg["har"].get("label_policy", "majority_non_null"))
    har_standardize = bool(cfg["har"].get("standardize_per_window", False))

    progress.stage("HAR: parsing raw streams, writing interim files, and creating windows")
    for dataset_name, parser, enabled in [
        ("pamap2", parse_pamap2_to_interim, cfg.get("pamap2", {}).get("enabled", True)),
        ("wisdm", parse_wisdm_to_interim, cfg.get("wisdm", {}).get("enabled", True)),
        ("mhealth", parse_mhealth_to_interim, cfg.get("mhealth", {}).get("enabled", True)),
    ]:
        if not enabled:
            continue
        dataset_raw = raw_dir / dataset_name
        if not dataset_raw.exists():
            _dataset_missing(dataset_name, dataset_requirements.get(dataset_name, True), mode, issues)
            continue
        files = _discover_har_files(dataset_name, dataset_raw)
        if not files:
            _dataset_missing(dataset_name, dataset_requirements.get(dataset_name, True), mode, issues)
            continue
        print(f"  HAR dataset={dataset_name}: {len(files)} source files", flush=True)
        for kind in ["pretrain", "supervised"]:
            stem = f"{dataset_name}_{kind}"
            out_dir = processed_dir / "har"
            if resume and _bundle_exists(out_dir, stem):
                print(f"  RESUME-SKIP HAR bundle already exists: {stem}", flush=True)
                manifest.extend(_bundle_manifest(out_dir, stem))
                continue
            all_arrays = []
            all_rows = []
            for idx, file in enumerate(files, start=1):
                print(f"    [{idx}/{len(files)}] {dataset_name}/{kind}: {file.name}", flush=True)
                interim_df = parser(file, schema)
                subject_frames = _iter_har_subject_frames(interim_df, default_subject_id=file.stem)
                for subject_id, subject_df in subject_frames:
                    subject_df = clean_and_resample_har(
                        subject_df,
                        channel_schema=schema,
                        target_hz=int(cfg["har"]["target_hz"]),
                        source_hz=cfg.get(dataset_name, {}).get("source_sampling_rate_hz"),
                        clip_zscore_at=cfg["har"].get("clip_zscore_at"),
                    )
                    subject_df = harmonize_har_labels(
                        subject_df,
                        dataset_name=dataset_name,
                        label_map=har_label_maps.get(dataset_name),
                        label_schema_name=cfg["har"].get("label_schema_name", DEFAULT_HAR_LABEL_SCHEMA_NAME),
                    )
                    interim_name = f"{file.stem}_{subject_id}_interim.csv" if len(subject_frames) > 1 else f"{file.stem}_interim.csv"
                    interim_path = interim_dir / "har" / dataset_name / interim_name
                    _write_interim_csv(interim_path, subject_df)
                    arr, rows = make_har_windows(
                        subject_df,
                        dataset_name=dataset_name,
                        subject_id=subject_id,
                        source_name=_har_source_id(dataset_raw, file),
                        sampling_rate_hz=int(cfg["har"]["target_hz"]),
                        channel_schema=schema,
                        kind=kind,
                        null_labels=null_labels,
                        label_policy=har_label_policy,
                        standardize_per_window=har_standardize,
                        pretrain_window_seconds=float(cfg["har"].get("pretrain_window_seconds", 10)),
                        pretrain_overlap_seconds=float(cfg["har"].get("pretrain_overlap_seconds", 0)),
                        supervised_window_seconds=float(cfg["har"].get("supervised_window_seconds", 5)),
                        supervised_overlap_seconds=float(cfg["har"].get("supervised_overlap_seconds", 2.5)),
                    )
                    if arr.size:
                        all_arrays.append(arr)
                    all_rows.extend(rows)
            bundle = np.concatenate(all_arrays, axis=0) if all_arrays else _empty_har_array(kind, len(schema), int(cfg["har"]["target_hz"]))
            manifest.extend(_bundle_dataset(out_dir, stem, bundle, all_rows))

    progress.stage("EEG: parsing EDF/CSV, extracting annotations, and creating event windows")
    eeg_raw = raw_dir / "eegmmidb"
    eeg_csv_files = sorted(eeg_raw.glob("*.csv")) if eeg_raw.exists() else []
    eeg_edf_files = sorted(eeg_raw.rglob("*.edf")) if eeg_raw.exists() else []
    if cfg.get("eeg", {}).get("enabled", True):
        if not (eeg_csv_files or eeg_edf_files):
            _dataset_missing("eegmmidb", dataset_requirements.get("eegmmidb", True), mode, issues)
        else:
            if resume and _bundle_exists(processed_dir / "eeg", "eegmmidb_events"):
                print("  RESUME-SKIP EEG bundle already exists: eegmmidb_events", flush=True)
                manifest.extend(_bundle_manifest(processed_dir / "eeg", "eegmmidb_events"))
            else:
                all_arrays = []
                all_rows = []
                file_list = eeg_edf_files if eeg_edf_files else eeg_csv_files
                target_sfreq = int(cfg["eeg"]["sampling_rate_hz"])
                print(f"  EEGMMIDB files discovered: {len(file_list)}", flush=True)
                for idx, file in enumerate(file_list, start=1):
                    print(f"    [{idx}/{len(file_list)}] EEG source: {file.name}", flush=True)
                    if file.suffix.lower() == ".edf":
                        signal, events, source_sfreq, channel_names = parse_eeg_edf_to_interim(
                            file,
                            bandpass_hz=tuple(cfg["eeg"].get("bandpass_hz", [1.0, 40.0])) if cfg["eeg"].get("bandpass_hz") else None,
                            notch_hz=cfg["eeg"].get("notch_hz"),
                            rereference=cfg["eeg"].get("rereference"),
                        )
                        subject_id, run_id = infer_subject_run_from_name(file.stem)
                    else:
                        signal, events = parse_eeg_csv_to_interim(file)
                        source_sfreq = float(cfg["eeg"]["sampling_rate_hz"])
                        channel_names = [f"ch_{i}" for i in range(signal.shape[1])]
                        subject_id, run_id = infer_subject_run_from_name(file.stem)
                    ensure_dir(interim_dir / "eeg")
                    allowed_events = [str(code) for code in cfg["eeg"]["event_codes"]]
                    if bool(cfg["eeg"].get("include_t0", False)) and "T0" not in allowed_events:
                        allowed_events = ["T0", *allowed_events]
                    events = events[events["event_code"].astype(str).isin(allowed_events)].copy()
                    output_sfreq = int(source_sfreq) if cfg["eeg"].get("keep_native_rate", True) and int(round(source_sfreq)) == target_sfreq else target_sfreq
                    if int(round(source_sfreq)) != output_sfreq:
                        signal = resample_array(signal, source_hz=float(source_sfreq), target_hz=output_sfreq)
                        if "event_onset_seconds" in events.columns:
                            events["event_onset"] = (events["event_onset_seconds"] * output_sfreq).round().astype(int)
                        else:
                            events["event_onset"] = (events["event_onset"].astype(float) * (output_sfreq / float(source_sfreq))).round().astype(int)
                    events.to_csv(interim_dir / "eeg" / f"{file.stem}_events.csv", index=False)
                    arr, rows = event_windows(
                        signal,
                        events,
                        output_sfreq,
                        int(cfg["eeg"]["window_seconds"]),
                        subject_id=subject_id,
                        run_id=run_id,
                        source_name=file.name,
                        channel_names=channel_names,
                        normalize_per_window=bool(cfg["eeg"].get("normalize_per_window", True)),
                        allowed_event_codes=set(allowed_events),
                    )
                    if arr.size:
                        all_arrays.append(arr)
                    all_rows.extend(rows)
                    _write_interim_summary(
                        interim_dir / "eeg" / f"{file.stem}_summary.json",
                        {
                            "source_file": file.name,
                            "n_samples": int(signal.shape[0]),
                            "n_channels": int(signal.shape[1]),
                            "sampling_rate_hz": int(output_sfreq),
                            "source_sampling_rate_hz": float(source_sfreq),
                            "channel_names": channel_names,
                            "event_count": int(len(events)),
                        },
                    )
                bundle = np.concatenate(all_arrays, axis=0) if all_arrays else np.empty((0, 64, target_sfreq * int(cfg["eeg"]["window_seconds"])), dtype=np.float32)
                manifest.extend(_bundle_dataset(processed_dir / "eeg", "eegmmidb_events", bundle, all_rows))

    progress.stage("ECG: parsing PTB-XL records, writing interim files, and preserving fold metadata")
    ecg_raw = raw_dir / "ptbxl"
    ecg_files = sorted(ecg_raw.glob("*.csv")) if ecg_raw.exists() else []
    ptbxl_meta_csv = ecg_raw / "ptbxl_database.csv"
    if cfg.get("ecg", {}).get("enabled", True):
        if not (ecg_files or ptbxl_meta_csv.exists()):
            _dataset_missing("ptbxl", dataset_requirements.get("ptbxl", True), mode, issues)
        else:
            if resume and _bundle_exists(processed_dir / "ecg", "ptbxl_records"):
                print("  RESUME-SKIP ECG bundle already exists: ptbxl_records", flush=True)
                manifest.extend(_bundle_manifest(processed_dir / "ecg", "ptbxl_records"))
            else:
                all_arrays = []
                all_rows = []
                if ptbxl_meta_csv.exists():
                    descriptors = load_ptbxl_metadata(ecg_raw, int(cfg["ecg"]["sampling_rate_hz"]))
                    max_records = cfg.get("ecg", {}).get("max_records")
                    if max_records is not None:
                        descriptors = descriptors[: int(max_records)]
                    print(f"  PTB-XL records discovered in official metadata: {len(descriptors)}", flush=True)
                    total = len(descriptors)
                    for idx, desc in enumerate(descriptors, start=1):
                        print(f"    [{idx}/{total}] PTB-XL record: {desc['record_id']}", flush=True)
                        record = load_ptbxl_record(desc["waveform_stem"])
                        ensure_dir(interim_dir / "ecg")
                        _write_interim_summary(
                            interim_dir / "ecg" / f"{desc['record_id']}_summary.json",
                            {
                                "record_id": desc["record_id"],
                                "patient_id": desc["patient_id"],
                                "n_channels": int(record.shape[0]),
                                "n_samples": int(record.shape[1]),
                                "sampling_rate_hz": int(desc["sampling_rate_hz"]),
                                "lead_names": desc.get("lead_names"),
                            },
                        )
                        row_array, row = record_to_processed(
                            record,
                            desc,
                            int(cfg["ecg"]["sampling_rate_hz"]),
                            int(cfg["ecg"]["holdout_fold"]),
                            normalize_per_record=bool(cfg["ecg"].get("normalize_per_record", True)),
                            remove_per_lead_mean=bool(cfg["ecg"].get("remove_per_lead_mean", True)),
                        )
                        all_arrays.append(row_array)
                        all_rows.append(row)
                else:
                    print(f"  PTB-XL fixture CSV files discovered: {len(ecg_files)}", flush=True)
                    for idx, file in enumerate(ecg_files, start=1):
                        print(f"    [{idx}/{len(ecg_files)}] PTB-XL fixture: {file.name}", flush=True)
                        record, meta = parse_ptbxl_csv_to_interim(file)
                        ensure_dir(interim_dir / "ecg")
                        _write_interim_summary(
                            interim_dir / "ecg" / f"{file.stem}_summary.json",
                            {
                                "record_id": meta["record_id"],
                                "patient_id": meta["patient_id"],
                                "n_channels": int(record.shape[0]),
                                "n_samples": int(record.shape[1]),
                                "sampling_rate_hz": int(meta["sampling_rate_hz"]),
                                "lead_names": meta.get("lead_names"),
                            },
                        )
                        row_array, row = record_to_processed(
                            record,
                            meta,
                            int(cfg["ecg"]["sampling_rate_hz"]),
                            int(cfg["ecg"]["holdout_fold"]),
                            normalize_per_record=bool(cfg["ecg"].get("normalize_per_record", True)),
                            remove_per_lead_mean=bool(cfg["ecg"].get("remove_per_lead_mean", True)),
                        )
                        all_arrays.append(row_array)
                        all_rows.append(row)
                _finalize_ptbxl_split_metadata(all_rows)
                ecg_samples = int(cfg["ecg"]["sampling_rate_hz"]) * 10
                bundle = np.concatenate(all_arrays, axis=0) if all_arrays else np.empty((0, 12, ecg_samples), dtype=np.float32)
                manifest.extend(_bundle_dataset(processed_dir / "ecg", "ptbxl_records", bundle, all_rows))

    progress.stage("Writing processed manifest and preprocess summary")
    write_json(reports_dir / "processed_manifest.json", manifest)
    summary_lines = ["# Preprocess summary", "", f"Mode: {mode}", f"Manifest entries: {len(manifest)}", ""]
    if issues:
        summary_lines.append("## Skipped or partial datasets")
        summary_lines.extend(f"- {msg}" for msg in issues)
        summary_lines.append("")
    summary_lines += [
        "## Resource-aware design",
        "",
        "- HAR processed file-by-file with harmonized labels and target-rate resampling.",
        "- EEG processed EDF-by-EDF and event-by-event; interim stores summaries plus parsed events rather than duplicating whole signals.",
        "- ECG processed record-by-record from metadata; interim stores record summaries instead of large duplicate tables.",
        "",
    ]
    (reports_dir / "preprocess_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    _write_compliance_outputs(reports_dir, manifest, issues, cfg)
    print(f"{status_text('OK', 'ok')} Processed manifest entries: {len(manifest)}", flush=True)
    return 0
