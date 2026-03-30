#!/usr/bin/env python3
"""Estimate current storage footprint and document RAM/chunking expectations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def bytes_in_tree(path: Path) -> int:
    """Return the total byte size of all files under a directory tree."""
    return sum(p.stat().st_size for p in path.glob("**/*") if p.is_file())


def _fmt(num: int) -> str:
    """Format a byte count into a compact human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num} B"


def _runtime_estimate(raw_b: int, interim_b: int, processed_b: int) -> tuple[str, str]:
    """Return a coarse low/high runtime estimate from current footprint."""
    total_gb = (raw_b + interim_b + processed_b) / (1024 ** 3)
    low = max(1.0, total_gb * 2.0)
    high = max(low + 1.0, total_gb * 6.0)
    return f"{low:.1f}", f"{high:.1f}"


def _dataset_sizes(base: Path) -> list[tuple[str, int]]:
    """Return top-level child directory sizes under a data tree."""
    if not base.exists():
        return []
    rows = []
    for child in sorted([p for p in base.iterdir() if p.is_dir()]):
        rows.append((child.name, bytes_in_tree(child)))
    return rows


def _load_stage_metrics(reports_dir: Path) -> dict[str, dict]:
    """Load optional measured stage metrics written by the root entrypoints."""
    out: dict[str, dict] = {}
    for stage in ["setup", "preprocess", "validate"]:
        path = reports_dir / f"{stage}_metrics.json"
        if path.exists():
            out[stage] = json.loads(path.read_text(encoding="utf-8"))
    return out


def main() -> int:
    """Build the resource-estimate markdown report from current outputs and stage metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-path", default="reports/resource_estimate.md")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    raw_b = bytes_in_tree(Path(args.raw_dir)) if Path(args.raw_dir).exists() else 0
    interim_b = bytes_in_tree(Path(args.interim_dir)) if Path(args.interim_dir).exists() else 0
    processed_b = bytes_in_tree(Path(args.processed_dir)) if Path(args.processed_dir).exists() else 0
    low_m, high_m = _runtime_estimate(raw_b, interim_b, processed_b)
    raw_rows = _dataset_sizes(Path(args.raw_dir))
    processed_rows = _dataset_sizes(Path(args.processed_dir))
    interim_rows = _dataset_sizes(Path(args.interim_dir))
    stage_metrics = _load_stage_metrics(Path(args.reports_dir))
    raw_table = "\n".join([f"| {name} | {size} | {_fmt(size)} |" for name, size in raw_rows]) or "| none | 0 | 0.00 B |"
    interim_table = "\n".join([f"| {name} | {size} | {_fmt(size)} |" for name, size in interim_rows]) or "| none | 0 | 0.00 B |"
    processed_table = "\n".join([f"| {name} | {size} | {_fmt(size)} |" for name, size in processed_rows]) or "| none | 0 | 0.00 B |"
    if stage_metrics:
        peak_lines = ["| Stage | Measured peak RAM | Source |", "|---|---|---|"]
        runtime_lines = ["| Stage | Latest runtime | Cumulative runtime | Attempts | Source |", "|---|---:|---:|---:|---|"]
        total_runtime = 0.0
        total_cumulative = 0.0
        for stage in ["setup", "preprocess", "validate"]:
            metrics = stage_metrics.get(stage)
            if metrics is None:
                continue
            peak = metrics.get("peak_rss_bytes_across_attempts") or metrics.get("peak_rss_bytes")
            runtime_s = float(metrics.get("duration_seconds", 0.0))
            cumulative_s = float(metrics.get("cumulative_duration_seconds", runtime_s))
            attempts = int(metrics.get("attempt_count", 1))
            total_runtime += runtime_s
            total_cumulative += cumulative_s
            peak_lines.append(f"| {stage} | {_fmt(int(peak)) if peak else 'Unavailable'} | `{stage}_metrics.json` |")
            runtime_lines.append(f"| {stage} | {runtime_s:.1f} s | {cumulative_s:.1f} s | {attempts} | `{stage}_metrics.json` |")
        runtime_lines.append(f"| **Total measured** | **{total_runtime / 60.0:.1f} min** | **{total_cumulative / 60.0:.1f} min** |  | combined stage metrics |")
        peak_section = "\n".join(peak_lines)
        runtime_section = "\n".join(runtime_lines)
        runtime_note = "These values come from measured stage metrics produced by the root entrypoints during real runs. Cumulative runtime captures repeated interrupted or resumed attempts for the same stage."
    else:
        peak_section = """| Stage | Expected peak RAM | Why |
|---|---|---|
| Download/setup | < 0.2 GB | streamed writes only |
| HAR preprocessing | < 1 GB | one source file plus windowing |
| EEG preprocessing | ~1 to 2 GB | one EDF plus event windows |
| ECG preprocessing | < 1 GB | record-by-record processing |
| Validation | < 0.5 GB | metadata-heavy rather than waveform-heavy |"""
        runtime_section = f"""| Stage | Estimate |
|---|---|
| Total preprocessing and validation after downloads complete | ~{low_m} to {high_m} minutes |"""
        runtime_note = "This remains a coarse heuristic based on current footprint and chunked streaming design, not a benchmark."
    text = f"""# Resource estimate

This report combines measured current on-disk footprint with runtime and memory information. Storage values below are measured from the current workspace. Runtime and RAM values are measured when stage metric JSON files are available; otherwise, heuristic expectations are shown.

## Raw storage by dataset

| Dataset | Bytes | Human-readable |
|---|---:|---:|
{raw_table}
| **Total** | **{raw_b}** | **{_fmt(raw_b)}** |

## Interim storage by modality

| Modality | Bytes | Human-readable |
|---|---:|---:|
{interim_table}
| **Total** | **{interim_b}** | **{_fmt(interim_b)}** |

## Processed storage by modality

| Modality | Bytes | Human-readable |
|---|---:|---:|
{processed_table}
| **Total** | **{processed_b}** | **{_fmt(processed_b)}** |

## Peak RAM expectations

{peak_section}

## Runtime estimate

{runtime_section}

{runtime_note}

## Chunking and streaming notes

- HTTP downloads are streamed to disk.
- EEGMMIDB EDF acquisition uses direct parallel HTTP and skips already-complete files.
- Archive extraction is done after download, not in-memory.
- UCI zip archives can be deleted after extraction to reduce redundancy.
- PTB-XL record acquisition is parallelized conservatively and skips already-downloaded files.
- Interim storage is intentionally lightweight for EEG and ECG to avoid duplicating raw waveform payloads.
- Processing uses modality-local iteration rather than whole-dataset loading.
"""
    Path(args.report_path).write_text(text, encoding="utf-8")
    print(args.report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
