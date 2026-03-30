# Validation report

## Required Validation Checks

| Check | Status | Evidence |
|---|---|---|
| Reproducibility: stage directories present and processed files agree with the manifest | PASS | stage_dirs_present=4/4, processed_files=16, manifest_entries=16, manifest_fresh=True, required_dataset_statuses={eegmmidb:downloaded, pamap2:downloaded, ptbxl:downloaded, wisdm:downloaded}, context_keys_missing=0, completed_stages=['preprocess', 'setup', 'validate'], missing_stages=[] |
| Sample pack status | PASS | sample metadata files detected: 8 |
| Resource awareness: storage totals reported here and peak RAM/runtime linked | PASS | raw=4275257185 B, interim=5002330500 B, processed=2847892020 B; see `reports/resource_estimate.md` |
| HAR harmonisation: datasets share the configured schema name | PASS | detected label schema names: ['har_unified_v1']; expected: har_unified_v1 |
| Array integrity: float32, no NaN/Inf, row counts, required metadata | PASS | checked 8 processed array bundles |
| HAR harmonisation: identical shared channel schema across datasets | PASS | detected schemas: ['acc_x|acc_y|acc_z|gyro_x|gyro_y|gyro_z'] |
| HAR harmonisation: supervised labels stay within the unified label set | PASS | unexpected labels: none |
| HAR window contract: 10 s pretrain, 5 s supervised, configured stride alignment | PASS | per-file HAR checks include length, stride alignment, and monotonic starts |
| HAR null/transient handling documented and fields present | PASS | null-label policy and original-label provenance checked in supervised HAR metadata |
| EEG annotation correctness | PASS | see EEG evidence section for configured event counts and run coverage |
| ECG fold correctness and leakage control | PASS | see ECG evidence section for split and stratified fold counts |

## HAR harmonization cross-check

- detected label schema names: ['har_unified_v1']
- expected label schema name: har_unified_v1
- detected shared channel schemas: ['acc_x|acc_y|acc_z|gyro_x|gyro_y|gyro_z']
- allowed supervised label set: ['cycling', 'household', 'lying', 'other', 'running', 'sitting', 'stairs', 'standing', 'walking']
- unexpected supervised labels: []

## HAR evidence

- mhealth_pretrain: windows=2427, subjects=10, labels=[]
- mhealth_supervised: windows=2996, subjects=10, labels=['cycling', 'household', 'lying', 'other', 'running', 'sitting', 'stairs', 'standing', 'walking']
- pamap2_pretrain: windows=3843, subjects=9, labels=[]
- pamap2_supervised: windows=11099, subjects=9, labels=['cycling', 'household', 'lying', 'other', 'running', 'sitting', 'stairs', 'standing', 'walking']
- wisdm_pretrain: windows=119936, subjects=51, labels=[]
- wisdm_supervised: windows=479777, subjects=51, labels=['other', 'running', 'sitting', 'standing', 'walking']

## EEG evidence

- eegmmidb_events: windows=4917, subjects=109, runs=3, events={'T1': 2479, 'T2': 2438}

### EEG Annotation Table

| Run | T1 events | T2 events | First T1 onset (s) |
|---|---:|---:|---:|
| 4 | 830 | 810 | 4.100 |
| 8 | 824 | 813 | 1.375 |
| 12 | 825 | 815 | 1.375 |

## ECG evidence

- ptbxl_records: records=21799, patients=18869, splits={'train': 19601, 'test': 2198}, strat_folds={1: 2175, 2: 2181, 3: 2192, 4: 2174, 5: 2174, 6: 2173, 7: 2176, 8: 2173, 9: 2183, 10: 2198}

### ECG Fold Table

| Split | Count | Percentage | strat_fold values |
|---|---:|---:|---|
| train | 19601 | 89.92% | 1,2,3,4,5,6,7,8,9 |
| test | 2198 | 10.08% | 10 |

**CV fold distribution (train only):**

| cv_fold | Count |
|---|---:|
| 1 | 2175 |
| 2 | 2181 |
| 3 | 2192 |
| 4 | 2174 |
| 5 | 2174 |
| 6 | 2173 |
| 7 | 2176 |
| 8 | 2173 |
| 9 | 2183 |

**Leakage check**: No patient appears in both train and test splits.

## Manifest checks

- processed manifest entries: 16

## Resource awareness

- raw storage bytes: 4275257185
- interim storage bytes: 5002330500
- processed storage bytes: 2847892020
- total processed arrays: 8
- chunking strategy: HAR file-by-file, EEG EDF-by-EDF, ECG record-by-record.
- full peak RAM and runtime details: see `reports/resource_estimate.md`.

## Submission sample checks

- sample metadata files: 8
- dataset eegmmidb: sample rows=100/100
- dataset mhealth: sample rows=100/100
- dataset pamap2: sample rows=100/100
- dataset ptbxl: sample rows=100/100
- dataset wisdm: sample rows=100/100

## Data integrity checks

### HAR
- PASS no NaN/Inf, fixed channel schema, and required HAR metadata fields
- PASS window lengths and stride alignment match the configured contract
### EEG
- PASS all retained EEG event windows passed shape and label checks
- PASS channel count is 64 across all retained windows
- PASS event onsets are monotonic within each subject/run
### ECG
- PASS PTB-XL rows passed fold and split checks
- PASS lead names are consistent with the configured lead order
- PASS sampling rate matches the configured ECG output rate

## Sample pack verification

| File | Expected Shape | Actual Shape | Status |
|---|---|---|---|
| eeg/eegmmidb/eegmmidb_events_sample.npz | (100, 64, 640) | (100, 64, 640) | PASS |
| har/mhealth/mhealth_pretrain_sample.npz | (50, 6, 200) | (50, 6, 200) | PASS |
| har/mhealth/mhealth_supervised_sample.npz | (50, 6, 100) | (50, 6, 100) | PASS |
| har/pamap2/pamap2_pretrain_sample.npz | (50, 6, 200) | (50, 6, 200) | PASS |
| har/pamap2/pamap2_supervised_sample.npz | (50, 6, 100) | (50, 6, 100) | PASS |
| ecg/ptbxl/ptbxl_records_sample.npz | (100, 12, 1000) | (100, 12, 1000) | PASS |
| har/wisdm/wisdm_pretrain_sample.npz | (50, 6, 200) | (50, 6, 200) | PASS |
| har/wisdm/wisdm_supervised_sample.npz | (50, 6, 100) | (50, 6, 100) | PASS |

## Array integrity and schema checks

- **ecg/ptbxl_records.npz**: shape=[21799, 12, 1000], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=21799, missing_required_columns=[]
- **eeg/eegmmidb_events.npz**: shape=[4917, 64, 640], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=4917, missing_required_columns=[]
- **har/mhealth_pretrain.npz**: shape=[2427, 6, 200], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=2427, missing_required_columns=[]
- **har/mhealth_supervised.npz**: shape=[2996, 6, 100], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=2996, missing_required_columns=[]
- **har/pamap2_pretrain.npz**: shape=[3843, 6, 200], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=3843, missing_required_columns=[]
- **har/pamap2_supervised.npz**: shape=[11099, 6, 100], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=11099, missing_required_columns=[]
- **har/wisdm_pretrain.npz**: shape=[119936, 6, 200], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=119936, missing_required_columns=[]
- **har/wisdm_supervised.npz**: shape=[479777, 6, 100], dtype=float32, has_nan=False, has_inf=False
  - metadata rows=479777, missing_required_columns=[]

## Overall status

PASS
