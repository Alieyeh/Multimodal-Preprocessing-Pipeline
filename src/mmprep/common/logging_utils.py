"""Utilities for lightweight run logging to both terminal and file."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO

from mmprep.common.console import status_text

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


class _TeeStream:
    """Write console output to the original stream and a logfile."""

    def __init__(self, console: TextIO, logfile: TextIO) -> None:
        self._console = console
        self._logfile = logfile

    def write(self, data: str) -> int:
        self._console.write(data)
        self._logfile.write(data)
        return len(data)

    def flush(self) -> None:
        self._console.flush()
        self._logfile.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._console, "isatty", lambda: False)())


def configure_logging(level: int = logging.INFO) -> None:
    """Configure stdlib logging for consistent timestamps."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


def _raw_tree_is_empty(repo_root: Path) -> bool:
    """Return whether `data/raw` is absent or contains no files/directories."""
    raw_dir = repo_root / "data" / "raw"
    if not raw_dir.exists():
        return True
    return not any(raw_dir.iterdir())


def _reset_stage_metrics(reports_dir: Path) -> None:
    """Remove existing stage-metric files when starting a fresh run from scratch."""
    for stage in ["setup", "preprocess", "validate"]:
        (reports_dir / f"{stage}_metrics.json").unlink(missing_ok=True)


@contextmanager
def stage_log(stage_name: str, repo_root: Path) -> Iterator[Path]:
    """Mirror stdout/stderr for one stage into a timestamped logfile.

    Inputs:
    - stage_name: short stage token such as `setup`, `preprocess`, or `validate`.
    - repo_root: repository root used to derive `data/logs`.

    Outputs:
    - Yields the created logfile path while teeing stdout and stderr to it.
    """
    logs_dir = repo_root / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{stage_name}_{stamp}.log"
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    fresh_setup_reset = stage_name == "setup" and _raw_tree_is_empty(repo_root)
    if fresh_setup_reset:
        _reset_stage_metrics(reports_dir)
    metrics_path = reports_dir / f"{stage_name}_metrics.json"
    existing_metrics = {}
    if metrics_path.exists():
        try:
            existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            existing_metrics = {}
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    started_at = time.perf_counter()
    peak_rss_bytes = 0
    stop_flag = threading.Event()

    def _sample_peak_memory() -> None:
        nonlocal peak_rss_bytes
        if psutil is None:
            return
        proc = psutil.Process()
        while not stop_flag.is_set():
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        continue
                peak_rss_bytes = max(peak_rss_bytes, int(rss))
            except Exception:
                pass
            stop_flag.wait(0.2)

    sampler = threading.Thread(target=_sample_peak_memory, daemon=True)
    if psutil is not None:
        sampler.start()
    with log_path.open("w", encoding="utf-8", newline="") as handle:
        sys.stdout = _TeeStream(original_stdout, handle)
        sys.stderr = _TeeStream(original_stderr, handle)
        try:
            print(f"{status_text('INFO', 'info')} [mmprep] stage={stage_name} log={log_path}", flush=True)
            if fresh_setup_reset:
                print(f"{status_text('INFO', 'info')} [mmprep] fresh setup detected; stage metrics reset", flush=True)
            yield log_path
        finally:
            stop_flag.set()
            if psutil is not None:
                sampler.join(timeout=1.0)
            duration_seconds = time.perf_counter() - started_at
            previous_attempts = int(existing_metrics.get("attempt_count", 0))
            previous_cumulative = float(existing_metrics.get("cumulative_duration_seconds", 0.0))
            previous_peak = existing_metrics.get("peak_rss_bytes_across_attempts")
            prior_peak_value = int(previous_peak) if previous_peak is not None else 0
            metrics = {
                "stage": stage_name,
                "log_path": str(log_path),
                "measured_at_local": datetime.now().isoformat(timespec="seconds"),
                "attempt_count": previous_attempts + 1,
                "duration_seconds": round(duration_seconds, 3),
                "cumulative_duration_seconds": round(previous_cumulative + duration_seconds, 3),
                "peak_rss_bytes": int(peak_rss_bytes) if peak_rss_bytes else None,
                "peak_rss_bytes_across_attempts": max(prior_peak_value, int(peak_rss_bytes or 0)) or None,
            }
            metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"{status_text('INFO', 'info')} [mmprep] stage={stage_name} metrics={metrics_path}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
