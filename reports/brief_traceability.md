# Brief Traceability

## Submission deliverables

- Wrapper script: `setup_data.py` plus `setup_data.sh` and `setup_data.ps1`
- Python preprocessing: `preprocess.py` and `src/mmprep/`
- Python validation: `validate_outputs.py` and `src/mmprep/validation.py`
- Concise brief-facing plan: `preprocessing_plan.md`
- Sample pack generator: `scripts/make_submission_sample.py`
- Sample pack manifest: `submission_sample/sample_pack_manifest.json`
- Validation report path: `reports/validation_report.md`
- Resource estimate path: `reports/resource_estimate.md`
- README with exact commands: `README.md`

## Section 5.1 Download and folder setup

- Clean folder structure: implemented in `setup_data.py:setup_folders`
- Scripted downloads only: shell/PowerShell wrappers call Python entrypoints, and the substantive download logic is implemented in `setup_data.py` for UCI, EEGMMIDB, and PTB-XL
- Source URLs, timestamps, versions: written to `reports/download_manifest.json`
- Clear failure behavior: strict mode raises on required dataset failures
- Timestamped setup logs: written automatically to `data/logs/`

## Section 5.2 Preprocessing plan

- Channel schema, sampling choices, windows, null-label handling, and resource expectations are documented concisely in `preprocessing_plan.md`

## Section 5.3 HAR

- Shared six-channel schema at 20 Hz: configured in `configs/default.yaml`
- Documented label harmonization: implemented in `src/mmprep/har/base.py`
- Required window definitions: implemented in `src/mmprep/common/windowing.py` and `src/mmprep/har/base.py`
- Subject and source provenance preserved: written in processed HAR metadata CSVs
- Representative sample pack remains dataset-level: HAR sample generation groups files under one folder per source dataset and splits the 100-row dataset budget across pretrain and supervised bundles in `scripts/make_submission_sample.py`
- mHealth bonus path: included and optional

## Section 5.4 EEG

- Default runs 4, 8, 12: configured in `configs/default.yaml`
- EDF+ parsing and event preservation: implemented in `src/mmprep/eeg/eegmmidb.py`
- 4-second T1/T2 windows: implemented in `event_windows`
- Fixed-shape EEG bundling: if a source EDF rate differs from the configured output rate, the signal is resampled before windowing in `src/mmprep/pipeline.py`
- Light preprocessing with justification: documented in `reports/scientific_justification.md`

## Section 5.5 ECG

- PTB-XL rate choice documented and configurable: `configs/default.yaml`
- Train/test and CV fold handling: implemented in `src/mmprep/ecg/ptbxl.py`
- Patient, record, label, lead, and fold metadata preserved in processed CSV outputs, with integer-like PTB-XL identifiers serialized without decimal suffixes

## Section 6 Processed output format

- Float32 `npz` arrays with shape `[N, C, T]`: implemented in `src/mmprep/common/io_utils.py`
- Required metadata columns: enforced in `src/mmprep/validation.py`
- Machine-readable processed manifest: written to `reports/processed_manifest.json`

## Section 7 Validation checks

- Reproducibility smoke tests: `tests/test_cli_smoke.py`
- Resume support for interrupted long preprocessing runs: `preprocess.py --resume` via `src/mmprep/cli.py` and `src/mmprep/pipeline.py`
- Timestamped run logs for setup, preprocess, and validate stages: `data/logs/`
- Reproducibility checks now cover setup artefacts, required submission docs, required dataset acquisition status, config/environment provenance, manifest coverage, and explicit completed-stage tracking for setup/preprocess/validate in `src/mmprep/validation.py` plus `src/mmprep/common/reproducibility.py`
- HAR harmonization checks: `src/mmprep/validation.py`
- Window definition checks: `src/mmprep/validation.py`
  fixed HAR window length, stride alignment, and monotonic start ordering are validated against the configured contract
- Null-label policy fields: validated in HAR outputs
- Array integrity checks: `summarize_npz`
- Provenance for leakage control: preserved across modalities, with PTB-XL patient overlap checked explicitly during validation
- EEG annotation checks: `src/mmprep/validation.py`
  validation is config-aware for allowed runs, event codes, output sampling rate, and window duration
- ECG fold and patient-split checks: `src/mmprep/validation.py`
- Submission sample checks: `src/mmprep/validation.py`
  sample files are validated against `submission_sample/sample_pack_manifest.json` for exact expected rows, exact expected shapes, and the 100-total-rows-per-dataset contract
- Resource awareness: `scripts/estimate_resources.py` and validation report section
- Validation CLI summary output: `validate_outputs.py` prints the report path, bundle counts, sample-pack counts, and final PASS/FAIL state
- Submission sample status reporting: validation reports when `submission_sample/` has not yet been generated rather than conflating that with processed-data corruption

## Section 9 Bonus

- mHealth integration: supported
- SSL handoff note: addressed in `reports/scientific_justification.md` and README framing
- Automated unit and smoke tests: included under `tests/`
