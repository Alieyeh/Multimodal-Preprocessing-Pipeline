# Preprocess summary

Mode: best-effort
Manifest entries: 16

## Resource-aware design

- HAR processed file-by-file with harmonized labels and target-rate resampling.
- EEG processed EDF-by-EDF and event-by-event; interim stores summaries plus parsed events rather than duplicating whole signals.
- ECG processed record-by-record from metadata; interim stores record summaries instead of large duplicate tables.
