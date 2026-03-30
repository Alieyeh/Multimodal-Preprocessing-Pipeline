# Clinical Dataset Adaptation

This pipeline is designed for easy adaptation to clinical and biobank datasets used in Parkinson's research. In that context, it is best viewed as a reproducible training-ready baseline for self-supervised learning and early supervised modelling, not as a finished disease-specific preprocessing standard for every downstream study.

## UK Biobank Accelerometry
- **Current capability**: HAR module handles 20 Hz IMU data from wrist-worn sensors
- **Adaptation needed**: Update parser to match UK Biobank CSV format (5-second epochs)
- **Effort**: ~1 day to add `src/mmprep/ukbb/accelerometry.py`

## PPMI Wearables
- **Current capability**: EEG module supports EDF+; ECG module supports 12-lead waveforms
- **Adaptation needed**: Add parser for PPMI's proprietary format if required
- **Effort**: Depends on format; modular structure allows drop-in replacements

## Electronic Health Records (CPRD, THIN)
- **Current capability**: Metadata-preserving design carries patient IDs and longitudinal provenance
- **Adaptation needed**: Add `src/mmprep/clinical/ehr.py` for longitudinal extraction
- **Effort**: ~3-5 days for an initial version; the validation framework can then be reused for QC

## Prescription Data
- **Current capability**: The manifest and metadata pattern already track provenance cleanly
- **Adaptation needed**: Add temporal alignment with clinical events
- **Effort**: Depends on schema; the existing pipeline structure already supports chunked processing

## Why This Matters for Parkinson's Research

The UK DRI's focus on the pre-diagnostic period requires:
1. **Longitudinal data integration** across wearables, EHR, and prescriptions
2. **Leakage-aware splitting** to avoid training on future data
3. **Reproducible preprocessing** across heterogeneous cohorts

This pipeline's design, especially metadata preservation, subject-level provenance, and modular parsers, directly addresses these requirements. For Parkinson's or other clinical modelling programmes, that makes it a strong starting point for cohort harmonisation and representation learning; later project phases would still likely revisit label definition, cohort-specific artifact handling, and task-specific quality-control rules once the exact clinical endpoint is fixed.
