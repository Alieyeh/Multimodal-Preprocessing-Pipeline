# Preprocessing Plan

Goal: produce reproducible fixed-shape `float32` arrays plus leakage-aware metadata for PAMAP2, WISDM, EEGMMIDB, PTB-XL, and optional mHealth.

Representations:
- HAR: six shared inertial channels `acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z` at 20 Hz.
- EEG: EEGMMIDB motor-imagery runs 4, 8, and 12, output at a fixed 160 Hz by default.
- ECG: PTB-XL records at 100 Hz by default, preserving patient and fold provenance.

Windowing:
- HAR pretraining: 10 s unlabeled windows, no overlap.
- HAR supervised: 5 s windows, 50% overlap, majority non-null label assignment.
- EEG: 4 s windows from T1/T2 onset by default; optional T0 retention is configurable.
- ECG: one fixed-length record tensor per PTB-XL sample; the held-out test fold is configurable via `ecg.holdout_fold` (default `10`).

HAR label harmonization:
- Unified labels: `lying, sitting, standing, walking, running, cycling, stairs, household, other`.
- `jogging -> running`; upstairs/downstairs are merged into `stairs`.
- Null/transient labels are excluded from supervised assignment, while `original_label` is retained for provenance.

Cleaning:
- HAR: numeric coercion, clipping, fill/interpolation, then resampling to 20 Hz.
- EEG: EDF+ parsing, optional rereference, optional 1 to 40 Hz band-pass, optional notch, fixed-rate output, per-window normalization.
- ECG: per-lead mean removal and per-record normalization.

Provenance and splits:
- HAR and EEG are emitted with subject/run/source provenance but without hard-coded train/validation/test splits.
- PTB-XL preserves patient, record, primary label, stratified fold, and derived train/test split metadata.

Resource notes:
- Downloads stream to disk and skip already-complete files.
- Processing is file-by-file or record-by-record, but HAR bundle assembly can still dominate RAM.
- Representative samples cap inspection artifacts at 100 rows per source dataset, with HAR budgets split across pretrain and supervised outputs.
