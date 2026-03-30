# Resource estimate

This report combines measured current on-disk footprint with runtime and memory information. Storage values below are measured from the current workspace. Runtime and RAM values are measured when stage metric JSON files are available; otherwise, heuristic expectations are shown.

## Raw storage by dataset

| Dataset | Bytes | Human-readable |
|---|---:|---:|
| eegmmidb | 834029636 | 795.39 MB |
| mhealth | 226819988 | 216.31 MB |
| pamap2 | 1732805748 | 1.61 GB |
| ptbxl | 542945564 | 517.79 MB |
| wisdm | 938656249 | 895.17 MB |
| **Total** | **4275257185** | **3.98 GB** |

## Interim storage by modality

| Modality | Bytes | Human-readable |
|---|---:|---:|
| ecg | 5733523 | 5.47 MB |
| eeg | 419910 | 410.07 KB |
| har | 4996177067 | 4.65 GB |
| **Total** | **5002330500** | **4.66 GB** |

## Processed storage by modality

| Modality | Bytes | Human-readable |
|---|---:|---:|
| ecg | 604890596 | 576.87 MB |
| eeg | 749751787 | 715.02 MB |
| har | 1493249637 | 1.39 GB |
| **Total** | **2847892020** | **2.65 GB** |

## Peak RAM expectations

| Stage | Measured peak RAM | Source |
|---|---|---|
| setup | 245.12 MB | `setup_metrics.json` |
| preprocess | 5.69 GB | `preprocess_metrics.json` |
| validate | 2.14 GB | `validate_metrics.json` |

## Runtime estimate

| Stage | Latest runtime | Cumulative runtime | Attempts | Source |
|---|---:|---:|---:|---|
| setup | 6147.7 s | 6147.7 s | 1 | `setup_metrics.json` |
| preprocess | 1398.2 s | 1398.2 s | 1 | `preprocess_metrics.json` |
| validate | 46.0 s | 46.0 s | 1 | `validate_metrics.json` |
| **Total measured** | **126.5 min** | **126.5 min** |  | combined stage metrics |

These values come from measured stage metrics produced by the root entrypoints during real runs. Cumulative runtime captures repeated interrupted or resumed attempts for the same stage.

## Chunking and streaming notes

- HTTP downloads are streamed to disk.
- EEGMMIDB EDF acquisition uses direct parallel HTTP and skips already-complete files.
- Archive extraction is done after download, not in-memory.
- UCI zip archives can be deleted after extraction to reduce redundancy.
- PTB-XL record acquisition is parallelized conservatively and skips already-downloaded files.
- Interim storage is intentionally lightweight for EEG and ECG to avoid duplicating raw waveform payloads.
- Processing uses modality-local iteration rather than whole-dataset loading.
