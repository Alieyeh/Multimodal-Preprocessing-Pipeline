# Scientific Justification

## Design Goal

The pipeline is designed for downstream self-supervised and supervised modelling, so the outputs remain close to the original time-series rather than collapsing signals into handcrafted aggregate features. The main design priorities are:

- scientifically defensible light preprocessing
- leakage-aware metadata preservation
- harmonised output contracts across datasets
- resource-aware execution on ordinary workstations

Across HAR, EEG, and ECG, the processed outputs remain signal-level tensors rather than handcrafted aggregate feature tables, so they stay suitable for downstream SSL and supervised modelling workflows.

## HAR Harmonisation

### Shared Channel Pairing Justification

The brief recommends a wrist/watch-like six-channel schema. This submission uses:

`acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z`

This is scientifically defensible because:

1. PAMAP2 wrist/hand IMU and WISDM watch signals are the most anatomically comparable streams available across the two mandatory HAR datasets.
2. Both sensor placements capture arm-swing and orientation changes strongly enough for locomotion and posture discrimination.
3. Using chest or ankle streams would introduce larger placement-driven distribution shift and weaken the harmonisation objective.
4. Accelerometer-plus-gyroscope pairing preserves both translational and rotational motion information, which is important for distinguishing activities with similar linear acceleration but different wrist orientation dynamics.

This choice is also consistent with wrist-worn wearable HAR literature, where upper-limb inertial sensing is a practical and well-established site for free-living activity recognition.

### Default HAR Label Schema

The default harmonised supervised label space is intentionally conservative. Activities that align cleanly are preserved; activities without a reliable counterpart are mapped to broader buckets rather than forcing artificial one-to-one correspondence. The default config records this schema in HAR metadata as `label_schema_name = har_unified_v1`, but both the mapping and schema name remain configurable if a reviewer wants to submit a different harmonisation contract.

| Unified Label | PAMAP2 Examples | WISDM Examples | mHealth Examples | Rationale |
|---|---|---|---|---|
| `lying` | `1` | N/A | `3` | Rest posture retained as a valid state |
| `sitting` | `2` | `sitting` | `2` | Stationary seated posture |
| `standing` | `3` | `standing` | `1` | Stationary upright posture |
| `walking` | `4` | `walking` | `4` | Direct locomotion match |
| `running` | `5` | `jogging` | `7` | High-intensity locomotion grouped together |
| `cycling` | `6` | N/A | `8` | Distinct cyclic activity preserved |
| `stairs` | `12`, `13` | `upstairs`, `downstairs` | `5`, `6` | Stair ascent/descent grouped when exact cross-dataset parity is weak |
| `household` | `16`, `17` | N/A | `10` | Non-locomotion activity bucket |
| `other` | unmatched activities | unmatched activities | unmatched activities | Explicit fallback rather than silent deletion |

### Null and Transient Label Handling

- PAMAP2 class `0` is treated as a null/transient label by default in supervised windowing.
- A window receives a supervised label by majority vote after excluding null/transient labels.
- Windows containing only null/transient labels are excluded from the supervised output.
- Original source labels are preserved in `original_label`, while the harmonised label is stored in `label_or_event`.

This rule is simple, reproducible, and avoids giving a supervised evaluation sample a meaningless class.

### HAR Sampling and Windowing

- Target rate: 20 Hz, as required by the brief
- Pretraining windows: 10 seconds, no overlap
- Supervised windows: 5 seconds, 50% overlap
- Optional per-window standardisation: enabled by default in config, applied after window extraction on a per-channel basis

Five-second labeled windows are long enough to stabilise human activity labels while still producing a practical number of samples. Ten-second unlabeled windows provide richer temporal context for representation learning and align well with modern subsequence-level time-series representation learning.

### Representative Sample-Pack Interpretation

The brief asks for a representative pack with 100 samples per dataset, while the HAR pipeline also needs to expose two distinct outputs per dataset: an unlabeled pretraining bundle and a labeled supervised-evaluation bundle. This submission therefore keeps the processed HAR outputs separate in `data/processed`, because merging 10-second unlabeled windows and 5-second labeled windows into one array would blur two scientifically different tasks and break the fixed-shape output contract.

Instead, the representative sample pack is organised per source dataset and the 100-row dataset budget is split across the corresponding HAR bundles. In the default full-data submission, this yields a balanced 50/50 split between pretraining and supervised HAR samples for each dataset. This interpretation stays faithful to the sampling requirement while preserving the methodological distinction between SSL-oriented and supervised-evaluation outputs. Each dataset folder in `submission_sample/` also includes a `sample_summary.json` file so the dataset-level sample total, and how it is distributed across the underlying sample artefacts, can be seen.

## EEG Preprocessing

### Why Runs 4, 8, and 12

These runs are the brief's required subset and correspond to a consistent motor-imagery setting with T1/T2 annotations. Restricting the default subset improves comparability and keeps the task focused on correct EDF ingestion and annotation-aligned preprocessing rather than broad task mixing.

### EEG Filtering Justification

- High-pass 1 Hz:
  removes slow drift and baseline wander while preserving the mu and beta rhythms commonly used in motor-imagery EEG
- Low-pass 40 Hz:
  keeps the main task-relevant sensorimotor content while reducing high-frequency noise and muscle contamination
- Notch filter:
  applied only when configured and only below Nyquist, because unnecessary notching can distort the spectrum
- Average reference:
  a standard light rereferencing choice that is widely used in EEG benchmarks and remains more defensible here than heavier spatial transforms

These settings are consistent with common motor-imagery EEG preprocessing practice, where preserving mu and beta rhythms while suppressing slow drift and obvious high-frequency artefact is more important than heavy subject-specific denoising.

### Why the Preprocessing Stays Light

No ICA, CSP, or aggressive subject-specific artifact modelling is performed in the default pipeline. Those methods can be useful for downstream modelling, but they also introduce stronger assumptions and more opportunities to overfit. For this brief, scientifically sound event parsing plus light preprocessing is the more sensible baseline.

### EEG Windowing Justification

The final EEG outputs are fixed 4-second windows beginning at T1 or T2 onset by default. That choice matches the brief and aligns with the event-related nature of motor imagery, where class-relevant information is temporally local to the cue. The config can optionally retain T0 rest windows as an extension, but the brief-facing default remains T1/T2 only.

For EEGMMIDB specifically, the default choice is to retain the native 160 Hz sampling rate because it already matches the configured fixed bundled output rate. Resampling is only used as a fallback when a source file does not match that rate and fixed-shape bundled outputs still need to be preserved.

For the final processed bundle, one configured output sampling rate is enforced so that all EEG windows share one inspectable fixed shape. The `keep_native_rate` flag therefore only preserves the native rate when it already matches the configured bundled rate; if a source EDF differs, the signal is resampled before window extraction. This is a practical requirement for the brief's fixed-shape output contract and is preferable to mixing incompatible window lengths in one bundled array.

## ECG Preprocessing

### Sampling Rate Trade-off: 100 Hz vs 500 Hz

| Factor | 100 Hz | 500 Hz |
|---|---|---|
| Samples per 10-second record | 1000 | 5000 |
| Storage and I/O cost | much lower | much higher |
| Broad morphology modelling | generally adequate | stronger temporal fidelity |
| High-frequency analysis | limited | better supported |

The default is 100 Hz because the brief explicitly allows either choice with justification, and for many morphology-focused representation-learning baselines, it preserves the major waveform structure while reducing storage roughly fivefold. The config still allows 500 Hz for users who need finer temporal detail.

### PTB-XL Fold Strategy Justification

PTB-XL already provides `strat_fold` values designed for leakage-aware benchmarking. This submission therefore uses:

1. Default test split: `strat_fold == holdout_fold` (submission default `10`)
2. Default training pool: configured non-holdout folds (submission default `[1..9]`)
3. Cross-validation metadata: `cv_fold = strat_fold` for non-test rows

This preserves the dataset's native patient-safe structure and avoids inventing a new split scheme unnecessarily. The submission default uses Fold 10 as holdout, but the config, processed metadata, and validation report all follow the configured holdout fold if that default is changed.

### ECG Signal Cleaning Justification

- Per-lead mean removal reduces baseline offsets
- Per-record normalization reduces scale heterogeneity across recordings
- The pipeline does not perform beat segmentation by default, because PTB-XL is already suitable for record-level preprocessing.

### ECG Metadata Preservation

The processed ECG metadata preserve the key fields needed for downstream supervised learning and leakage-aware evaluation: patient identifier, record identifier, label, sampling rate, lead names, holdout split assignment, and cross-validation fold metadata. This keeps the waveform outputs directly traceable back to the source records while supporting later patient-safe model development.

### PTB-XL Label Caveat

PTB-XL is natively multi-label, but the current processed metadata reduce each record to its highest-weight SCP code. That is acceptable for a lightweight pipeline and keeps the output contract simple, but it should be understood as a simplification rather than a fully faithful representation of the annotation space.

## Bonus: mHealth Integration

mHealth is included as an optional third HAR source. It is resampled to the same 20 Hz target and mapped into the same six-channel inertial schema where possible. Label `0` or equivalent null labels are handled consistently with the rest of the HAR pipeline: preserved in provenance, excluded from supervised majority-vote label assignment, and documented explicitly.

## Clinical Translation Relevance

For downstream machine-learning use, the current pipeline should be thought of as a strong training-ready baseline rather than a fully task-optimized clinical preprocessing stack. It is well suited to self-supervised pretraining, representation learning, and baseline supervised modelling because it preserves waveform or window-level structure, leakage-aware provenance, and fixed inspectable tensor contracts.

The same design principles are directly transferable to clinical and population-health cohorts such as PPMI, UK Biobank, and All of Us:

| Design principle | Clinical or biobank relevance |
|---|---|
| Subject-level provenance | prevents leakage across repeated visits or longitudinal follow-up |
| Resumable downloads and manifests | practical for large controlled-access transfers and audit trails |
| Metadata-preserving preprocessing | retains diagnostic, medication, and visit-level context for later modelling |
| Modular parser layout | allows new data families such as EHR, wearables, imaging-derived phenotypes, or omics to be added without changing the output contract |
| Validation before modelling | catches corruption, split leakage, and schema drift before downstream analysis |

This is especially relevant for Parkinson's research, where repeated measures, multimodal phenotyping, and leakage-aware patient-level splitting are central. If this pipeline were pushed into a higher-stakes disease-modelling study rather than a preprocessing-focused baseline, the first scientific areas to revisit would be richer PTB-XL label handling, more explicit EEG artifact strategy, and whether the current HAR harmonization and `other` bucket are optimal for the precise downstream question.

## Literature Grounding

The scientific defaults above are aligned with representative literature and source documentation:

1. WISDM Smartphone and Smartwatch Activity and Biometrics Dataset. UCI Machine Learning Repository dataset page and bundled dataset description PDF. Used here as the primary source for watch accelerometer and gyroscope structure, 20 Hz sampling, 51 subjects, and activity-code definitions.
2. Tangermann, Michael et al. "Review of the BCI Competition IV." *Frontiers in Neuroscience* 6:55, 2012. Representative reference for motor-imagery EEG preprocessing context.
3. Goldberger, Ary L. et al. "PhysioBank, PhysioToolkit, and PhysioNet." *Circulation* 101(23), 2000. Background reference for PhysioNet-hosted acquisition workflows.
4. Wagner, Patrick et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154, 2020. Supports the native `strat_fold` split strategy.
5. Yue, Zhihan et al. "TS2Vec: Towards Universal Representation of Time Series." *AAAI*, 2022. Representative reference for context-rich subsequence learning in time-series SSL.

## Downstream SSL Handoff

The processed outputs are ready for several common self-supervised workflows:

1. Contrastive learning on HAR or EEG windows using same-subject or same-session positives
2. Masked-signal reconstruction for ECG and EEG
3. Time-domain augmentation pipelines using scaling, jitter, masking, and time-warp transforms
4. Leakage-aware train/validation/test construction using preserved subject or patient identifiers and fold metadata

A natural next step is a PyTorch dataset layer that loads `npz` arrays plus CSV metadata and returns `(X, metadata)` pairs for SSL pretraining. The current outputs are therefore appropriate for SSL handoff and sensible first-pass supervised experiments, while still leaving room for later task-specific refinement when the modelling objective becomes more clinically specific.

## Resource Awareness Is Part of Scientific Quality

Streaming downloads, resumable transfers, chunked processing, and restrained interim duplication are not just engineering conveniences. They materially improve reproducibility by making it more realistic to rerun the full pipeline on ordinary hardware without silently changing methodology to fit machine limits.

The preprocessing stage also supports bundle-level resume for long interrupted runs, which is especially important when full public-data preprocessing is used to generate the final submission artefacts.
