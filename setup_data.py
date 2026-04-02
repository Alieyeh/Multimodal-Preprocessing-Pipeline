#!/usr/bin/env python3
"""Cross-platform setup entrypoint for folder creation and dataset acquisition.

This wrapper is intentionally responsible for only three things:
1. Creating the clean brief-facing directory layout.
2. Downloading or simulating raw datasets.
3. Writing a machine-readable download manifest and setup summary.

It does not perform substantive preprocessing.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import os
import re
import threading
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    import requests
except Exception:  # pragma: no cover - optional dependency fallback
    requests = None

import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmprep.common.io_utils import ensure_dir, write_json
from mmprep.common.console import status_text
from mmprep.common.logging_utils import stage_log
from mmprep.common.progress import ProgressPrinter
from mmprep.common.reproducibility import update_reproducibility_context


@dataclass
class DownloadRecord:
    dataset_name: str
    source_url: str
    downloaded_at_utc: str
    local_path: str
    status: str
    version: str | None = None
    notes: str | None = None
    file_count: int | None = None
    bytes_downloaded: int | None = None


_METADATA_FILES = ("ptbxl_database.csv", "scp_statements.csv")
_HTTP_SESSION = None
_HTTP_DEBUG_LOCK = threading.Lock()
_HTTP_DEBUG_PATH: Path | None = None


def _tqdm_stream():
    """Send tqdm bars to the original console stream when stdout/stderr are tee-wrapped."""
    return getattr(sys.stderr, "_console", sys.stderr)


def _get_http_debug_path() -> Path:
    """Return the setup HTTP debug log path under data/logs."""
    global _HTTP_DEBUG_PATH
    if _HTTP_DEBUG_PATH is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logs_dir = REPO_ROOT / "data" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        _HTTP_DEBUG_PATH = logs_dir / f"setup_http_headers_{stamp}.jsonl"
    return _HTTP_DEBUG_PATH


def _headers_to_dict(headers: object) -> dict[str, str]:
    """Convert headers into a plain JSON-serialisable dictionary."""
    if headers is None:
        return {}
    try:
        return {str(k): str(v) for k, v in headers.items()}
    except Exception:
        return {}


def _content_range_total(headers: object) -> int | None:
    """Extract the total byte size from a Content-Range header when present."""
    try:
        value = headers.get("Content-Range") if headers is not None else None
    except Exception:
        value = None
    if not value:
        return None
    match = re.search(r"bytes\s+\d+-\d+/(\d+|\*)", str(value), flags=re.IGNORECASE)
    if not match:
        return None
    total = match.group(1)
    return int(total) if total.isdigit() else None


def _log_http_debug(event: dict) -> None:
    """Append one HTTP diagnostic event to the setup header log."""
    path = _get_http_debug_path()
    payload = {"logged_at_utc": _now(), **event}
    with _HTTP_DEBUG_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse setup-stage command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--include-mhealth", dest="include_mhealth", action="store_true", default=True, help="Include optional mHealth bonus dataset during setup.")
    parser.add_argument("--exclude-mhealth", dest="include_mhealth", action="store_false", help="Skip optional mHealth bonus dataset during setup.")
    parser.add_argument("--simulate-downloads", action="store_true", help="Create marker files instead of live downloads.")
    parser.add_argument("--mode", choices=["strict", "best-effort"], default=None, help="Override config setup mode.")
    parser.add_argument("--workers", type=int, default=None, help="Override config worker count for HTTP downloads.")
    return parser.parse_args()


def _now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _infer_dataset_version_label(url: str, configured_version: str | None = None) -> str | None:
    """Infer a stable source version or identifier when one is not configured.

    Preference order:
    1. Explicit configured version.
    2. PhysioNet semantic version from `/files/<slug>/<version>/`.
    3. UCI stable public dataset identifier from `/public/<id>/`.
    """
    if configured_version not in {None, ""}:
        return str(configured_version)
    path = urlparse(url).path.strip("/")
    physionet_match = re.search(r"/files/[^/]+/(\d+\.\d+(?:\.\d+)?)/?$", "/" + path)
    if physionet_match:
        return physionet_match.group(1)
    uci_match = re.search(r"/public/(\d+)/", "/" + path)
    if uci_match:
        return f"uci-public-{uci_match.group(1)}"
    return None


def _touch_marker(dir_path: Path, name: str, content: str) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / name).write_text(content, encoding="utf-8")


def setup_folders(project_root: Path) -> dict[str, Path]:
    """Create the full brief-facing folder layout."""
    return {
        "raw": ensure_dir(project_root / "data" / "raw"),
        "interim": ensure_dir(project_root / "data" / "interim"),
        "processed": ensure_dir(project_root / "data" / "processed"),
        "reports": ensure_dir(project_root / "reports"),
        "submission_sample": ensure_dir(project_root / "submission_sample"),
        "logs": ensure_dir(project_root / "data" / "logs"),
    }


def _load_config(project_root: Path, config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = project_root / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _get_http_session():
    """Create a pooled HTTP session when requests is available."""
    global _HTTP_SESSION
    if requests is None:
        return None
    if _HTTP_SESSION is None:
        _HTTP_SESSION = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=0)
        _HTTP_SESSION.mount("http://", adapter)
        _HTTP_SESSION.mount("https://", adapter)
        _HTTP_SESSION.headers.update({"User-Agent": "mmprep/1.0"})
    return _HTTP_SESSION


def _remote_size(url: str, timeout: int) -> int | None:
    """Fetch remote content length when the server advertises it.

    Inputs:
    - url: remote HTTP or HTTPS URL.
    - timeout: request timeout in seconds.

    Outputs:
    - Integer byte size when available, otherwise None.
    """
    session = _get_http_session()
    if session is not None:
        try:
            response = session.head(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            value = response.headers.get("Content-Length")
            _log_http_debug({
                "phase": "remote_size",
                "method": "HEAD",
                "url": url,
                "status_code": int(response.status_code),
                "final_url": str(getattr(response, "url", url)),
                "content_length": value,
                "headers": _headers_to_dict(response.headers),
            })
            if value:
                return int(value)
        except Exception as exc:
            _log_http_debug({
                "phase": "remote_size",
                "method": "HEAD",
                "url": url,
                "status": "error",
                "error": repr(exc),
            })
        try:
            probe_headers = {"User-Agent": "mmprep/1.0", "Range": "bytes=0-0", "Accept-Encoding": "identity"}
            response = session.get(url, timeout=timeout, allow_redirects=True, headers=probe_headers, stream=True)
            response.raise_for_status()
            total = _content_range_total(response.headers)
            _log_http_debug({
                "phase": "remote_size_probe",
                "method": "GET",
                "url": url,
                "status_code": int(response.status_code),
                "final_url": str(getattr(response, "url", url)),
                "request_headers": probe_headers,
                "headers": _headers_to_dict(response.headers),
                "content_range_total": total,
            })
            response.close()
            return total
        except Exception as exc:
            _log_http_debug({
                "phase": "remote_size_probe",
                "method": "GET",
                "url": url,
                "status": "error",
                "error": repr(exc),
            })
            return None
    req = Request(url, method="HEAD", headers={"User-Agent": "mmprep/1.0"})
    try:
        with urlopen(req, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            _log_http_debug({
                "phase": "remote_size",
                "method": "HEAD",
                "url": url,
                "status_code": int(getattr(response, "status", 0) or 0),
                "final_url": str(getattr(response, "url", url)),
                "content_length": value,
                "headers": _headers_to_dict(response.headers),
            })
            if value:
                return int(value)
    except Exception as exc:
        _log_http_debug({
            "phase": "remote_size",
            "method": "HEAD",
            "url": url,
            "status": "error",
            "error": repr(exc),
        })
    try:
        probe_headers = {"User-Agent": "mmprep/1.0", "Range": "bytes=0-0", "Accept-Encoding": "identity"}
        req = Request(url, headers=probe_headers)
        with urlopen(req, timeout=timeout) as response:
            total = _content_range_total(response.headers)
            _log_http_debug({
                "phase": "remote_size_probe",
                "method": "GET",
                "url": url,
                "status_code": int(getattr(response, "status", 0) or 0),
                "final_url": str(getattr(response, "url", url)),
                "request_headers": probe_headers,
                "headers": _headers_to_dict(response.headers),
                "content_range_total": total,
            })
            return total
    except Exception as exc:
        _log_http_debug({
            "phase": "remote_size_probe",
            "method": "GET",
            "url": url,
            "status": "error",
            "error": repr(exc),
        })
        return None


def _sha256_prefix(path: Path, prefix_bytes: int = 1024 * 1024) -> str:
    """Hash the leading bytes of a file for lightweight provenance.

    Inputs:
    - path: file to fingerprint.
    - prefix_bytes: maximum number of bytes to hash.

    Outputs:
    - Short hexadecimal digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(prefix_bytes))
    return digest.hexdigest()[:16]


def _zip_integrity_error(path: Path) -> str | None:
    """Return an integrity error string for an invalid zip, otherwise None."""
    if not path.exists():
        return f"{path.name} is missing"
    if not zipfile.is_zipfile(path):
        return f"{path.name} is not recognised as a valid zip archive"
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.infolist()
    except Exception as exc:
        return f"{path.name} failed zip validation: {exc}"
    return None


def _stream_http(url: str, destination: Path, timeout: int, retries: int, show_progress: bool = True) -> dict:
    """Stream an HTTP download to disk with optional live CLI progress.

    Inputs:
    - url: remote file URL.
    - destination: local output file path.
    - timeout: per-request timeout in seconds.
    - retries: number of attempts before failing.
    - show_progress: whether to render a per-file tqdm byte bar.

    Outputs:
    - Writes the file to disk atomically via a temporary .part file.
    - Returns byte-count provenance for manifesting and logging.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    temp_path = destination.with_suffix(destination.suffix + ".part")
    remote_size = _remote_size(url, timeout=timeout)
    session = _get_http_session()
    for attempt in range(1, retries + 1):
        try:
            if destination.exists() and destination.stat().st_size > 0:
                local_size = destination.stat().st_size
                if remote_size is not None and local_size == remote_size:
                    print(f"      {status_text('OK', 'ok')} SKIP {destination.name}: local size matches remote ({local_size} bytes)", flush=True)
                    return {
                        "status": "skipped_existing",
                        "bytes": local_size,
                        "remote_bytes": remote_size,
                        "fingerprint": _sha256_prefix(destination),
                    }
            resume_from = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"User-Agent": "mmprep/0.9"}
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
            mode = "ab" if resume_from > 0 else "wb"
            if session is not None:
                request_headers = {"Accept-Encoding": "identity", **headers}
                with session.get(url, stream=True, timeout=timeout, headers=request_headers) as response:
                    _log_http_debug({
                        "phase": "download",
                        "method": "GET",
                        "url": url,
                        "destination": str(destination),
                        "attempt": int(attempt),
                        "resume_from": int(resume_from),
                        "status_code": int(response.status_code),
                        "final_url": str(getattr(response, "url", url)),
                        "request_headers": request_headers,
                        "response_headers": _headers_to_dict(response.headers),
                        "content_length": response.headers.get("Content-Length"),
                    })
                    if show_progress and remote_size is None and response.headers.get("Content-Length") in (None, ""):
                        if response.headers.get("Transfer-Encoding", "").lower() == "chunked":
                            print(
                                f"      {status_text('INFO', 'info')} remote size unavailable for {destination.name}: "
                                "server returned chunked transfer without Content-Length; progress will show downloaded bytes only.",
                                flush=True,
                            )
                    if response.status_code == 416 and remote_size is not None and temp_path.exists() and temp_path.stat().st_size == remote_size:
                        temp_path.replace(destination)
                        return {
                            "status": "downloaded",
                            "bytes": destination.stat().st_size,
                            "remote_bytes": remote_size,
                            "fingerprint": _sha256_prefix(destination),
                        }
                    response.raise_for_status()
                    if resume_from > 0 and response.status_code == 200:
                        print(
                            f"      {status_text('WARNING', 'warn')} server ignored resume request for {destination.name}; restarting full download from byte 0.",
                            flush=True,
                        )
                        temp_path.unlink(missing_ok=True)
                        resume_from = 0
                        mode = "wb"
                    response_size = int(response.headers.get("Content-Length", "0") or 0)
                    total = remote_size
                    if total is None and response_size > 0:
                        total = resume_from + response_size if response.status_code == 206 else response_size
                    progress = tqdm(
                        total=total if total and total > 0 else None,
                        initial=resume_from if total and total > 0 else 0,
                        desc=destination.name,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        dynamic_ncols=True,
                        leave=False,
                        file=_tqdm_stream(),
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        disable=not show_progress,
                    )
                    with temp_path.open(mode) as out_f:
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if not chunk:
                                continue
                            out_f.write(chunk)
                            progress.update(len(chunk))
                    progress.close()
            else:
                req = Request(url, headers=headers)
                with urlopen(req, timeout=timeout) as response, temp_path.open(mode) as out_f:
                    status_code = getattr(response, "status", None)
                    _log_http_debug({
                        "phase": "download",
                        "method": "GET",
                        "url": url,
                        "destination": str(destination),
                        "attempt": int(attempt),
                        "resume_from": int(resume_from),
                        "status_code": int(status_code or 0),
                        "final_url": str(getattr(response, "url", url)),
                        "request_headers": headers,
                        "response_headers": _headers_to_dict(response.headers),
                        "content_length": response.headers.get("Content-Length"),
                    })
                    if show_progress and remote_size is None and response.headers.get("Content-Length") in (None, ""):
                        if response.headers.get("Transfer-Encoding", "").lower() == "chunked":
                            print(
                                f"      {status_text('INFO', 'info')} remote size unavailable for {destination.name}: "
                                "server returned chunked transfer without Content-Length; progress will show downloaded bytes only.",
                                flush=True,
                            )
                    if resume_from > 0 and status_code != 206:
                        print(
                            f"      {status_text('WARNING', 'warn')} server ignored resume request for {destination.name}; restarting full download from byte 0.",
                            flush=True,
                        )
                        out_f.seek(0)
                        out_f.truncate()
                        resume_from = 0
                    response_size = int(response.headers.get("Content-Length", "0") or 0)
                    total = remote_size
                    if total is None and response_size > 0:
                        total = resume_from + response_size if status_code == 206 else response_size
                    progress = None
                    if show_progress:
                        progress = tqdm(
                            total=total if total and total > 0 else None,
                            initial=resume_from if total and total > 0 else 0,
                            desc=destination.name,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            dynamic_ncols=True,
                            leave=False,
                            file=_tqdm_stream(),
                            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        )
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
                    if progress is not None:
                        progress.close()
            temp_path.replace(destination)
            return {
                "status": "downloaded",
                "bytes": destination.stat().st_size,
                "remote_bytes": remote_size,
                "fingerprint": _sha256_prefix(destination),
            }
        except Exception as exc:  # pragma: no cover - network dependent
            last_exc = exc
            _log_http_debug({
                "phase": "download",
                "method": "GET",
                "url": url,
                "destination": str(destination),
                "attempt": int(attempt),
                "status": "error",
                "error": repr(exc),
            })
            if temp_path.exists():
                temp_path.unlink()
            if attempt < retries:
                sleep_s = min(2 * attempt, 10)
                print(f"      {status_text('WARNING', 'warn')} retry {attempt}/{retries - 1} for {destination.name} after error: {exc}", flush=True)
                time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


def _extract_zip_recursive(archive_path: Path, output_dir: Path, remove_archives_after_extract: bool = False) -> int:
    """Extract a zip and recursively extract any nested zips discovered.

    Inputs:
    - archive_path: downloaded archive path.
    - output_dir: extraction root directory.
    - remove_archives_after_extract: whether to delete archives after successful extraction.

    Outputs:
    - Count of extracted members across the main archive and nested archives.
    """
    extracted = 0
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)
        extracted += len(zf.namelist())
    if remove_archives_after_extract:
        archive_path.unlink(missing_ok=True)
    seen = {archive_path.resolve()}
    while True:
        nested = sorted(
            path.resolve()
            for path in output_dir.rglob("*.zip")
            if path.resolve() not in seen and path.exists()
        )
        if not nested:
            break
        for nested_zip in nested:
            seen.add(nested_zip)
            nested_out = nested_zip.with_suffix("")
            nested_out.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(nested_zip, "r") as zf:
                zf.extractall(nested_out)
                extracted += len(zf.namelist())
            if remove_archives_after_extract:
                nested_zip.unlink(missing_ok=True)
    return extracted


def _download_uci_dataset(name: str, url: str, out_dir: Path, timeout: int, retries: int, remove_archives_after_extract: bool = False, dataset_version: str | None = None) -> DownloadRecord:
    """Download and extract a UCI archive-backed dataset.

    Inputs:
    - name: dataset name.
    - url: archive URL.
    - out_dir: dataset raw directory.
    - timeout: HTTP timeout in seconds.
    - retries: number of attempts.
    - remove_archives_after_extract: whether to delete the archive after extraction.

    Outputs:
    - Download manifest record describing the raw dataset.
    """
    archive_name = Path(urlparse(url).path).name or f"{name}.zip"
    archive_path = out_dir / archive_name
    temp_archive_path = archive_path.with_suffix(archive_path.suffix + ".part")
    extraction_marker = out_dir / "EXTRACTION_COMPLETE.json"
    resolved_version = _infer_dataset_version_label(url, dataset_version)
    if extraction_marker.exists():
        payload = json.loads(extraction_marker.read_text(encoding="utf-8"))
        return DownloadRecord(
            dataset_name=name,
            source_url=url,
            downloaded_at_utc=_now(),
            local_path=str(out_dir),
            status="skipped_existing",
            version=payload.get("dataset_version", resolved_version),
            notes="Archive extraction marker already present; reusing existing raw dataset tree.",
            file_count=int(payload.get("file_count", 0)),
        )
    transfer = None
    for archive_attempt in range(1, max(1, retries) + 1):
        if archive_path.exists():
            archive_error = _zip_integrity_error(archive_path)
            if archive_error is None:
                print(f"    {status_text('INFO', 'info')} Reusing validated archive: {archive_name}", flush=True)
                transfer = {
                    "status": "skipped_existing",
                    "bytes": archive_path.stat().st_size,
                    "remote_bytes": None,
                    "fingerprint": _sha256_prefix(archive_path),
                }
                break
            print(
                f"    {status_text('WARNING', 'warn')} existing archive for {name} is invalid ({archive_error}); deleting and downloading a clean copy.",
                flush=True,
            )
            archive_path.unlink(missing_ok=True)
            temp_archive_path.unlink(missing_ok=True)
        print(f"    {status_text('INFO', 'info')} Downloading archive: {archive_name}", flush=True)
        transfer = _stream_http(url, archive_path, timeout=timeout, retries=retries, show_progress=True)
        archive_error = _zip_integrity_error(archive_path)
        if archive_error is None:
            break
        archive_path.unlink(missing_ok=True)
        temp_archive_path.unlink(missing_ok=True)
        if archive_attempt >= max(1, retries):
            raise zipfile.BadZipFile(
                f"Downloaded archive for {name} failed integrity validation ({archive_error}). "
                "The local copy was removed so the next run starts from a clean slate."
            )
        print(
            f"    {status_text('WARNING', 'warn')} downloaded archive for {name} failed integrity validation ({archive_error}); retrying from scratch.",
            flush=True,
        )
    assert transfer is not None
    print(f"    {status_text('INFO', 'info')} Extracting archive into {out_dir}", flush=True)
    extracted_count = _extract_zip_recursive(archive_path, out_dir, remove_archives_after_extract=remove_archives_after_extract)
    write_json(
        extraction_marker,
        {
            "archive_name": archive_name,
            "dataset_version": resolved_version,
            "download_status": transfer["status"],
            "downloaded_bytes": transfer["bytes"],
            "remote_bytes": transfer["remote_bytes"],
            "file_count": extracted_count,
            "fingerprint": transfer["fingerprint"],
            "remove_archives_after_extract": remove_archives_after_extract,
        },
    )
    _touch_marker(out_dir, "DOWNLOAD_COMPLETE.txt", f"Dataset={name}\nURL={url}\n")
    return DownloadRecord(
        dataset_name=name,
        source_url=url,
        downloaded_at_utc=_now(),
        local_path=str(out_dir),
        status=transfer["status"],
        version=resolved_version,
        notes="Archive downloaded and recursively extracted (handles nested zip payloads where present, e.g. WISDM, PAMAP2).",
        file_count=extracted_count,
        bytes_downloaded=transfer["bytes"],
    )


def _iter_eegmmidb_targets(runs: list[int], subjects: list[int] | None) -> Iterable[tuple[int, int, str]]:
    """Yield EEGMMIDB subject/run download targets."""
    subject_list = subjects or list(range(1, 110))
    for subject in subject_list:
        for run in runs:
            rel = f"S{int(subject):03d}/S{int(subject):03d}R{int(run):02d}.edf"
            yield int(subject), int(run), rel


def _download_eegmmidb(out_dir: Path, source_url: str, runs: list[int], subjects: list[int] | None, timeout: int, retries: int, workers: int) -> DownloadRecord:
    """Download EEGMMIDB EDF files directly from PhysioNet.

    Inputs:
    - out_dir: dataset raw directory.
    - source_url: PhysioNet base URL ending in the dataset version directory.
    - runs: requested EEGMMIDB runs.
    - subjects: requested subject subset or None for all subjects.
    - timeout: HTTP timeout in seconds.
    - retries: number of attempts.
    - workers: parallel download worker count.

    Outputs:
    - Download manifest record describing the EDF acquisition.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = source_url.rstrip("/") + "/"
    targets = list(_iter_eegmmidb_targets(runs=runs, subjects=subjects))
    total = len(targets)
    print(f"    {status_text('INFO', 'info')} Downloading EEGMMIDB EDF files for {total} subject/run pairs with {max(1, workers)} workers", flush=True)

    def _task(target: tuple[int, int, str]) -> dict:
        subject, run, rel = target
        dest = out_dir / rel
        transfer = _stream_http(base_url + rel, dest, timeout=timeout, retries=retries, show_progress=(max(1, workers) == 1))
        return {"subject": subject, "run": run, "path": str(dest), **transfer}

    results: list[dict] = []
    if max(1, workers) == 1:
        bar = tqdm(targets, total=total, desc="EEGMMIDB EDF", unit="file", dynamic_ncols=True, file=_tqdm_stream(), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for target in bar:
            _, _, rel = target
            bar.set_postfix_str(rel)
            results.append(_task(target))
        bar.close()
    else:
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            future_map = {ex.submit(_task, target): target for target in targets}
            for future in tqdm(cf.as_completed(future_map), total=total, desc="EEGMMIDB EDF", unit="file", dynamic_ncols=True, file=_tqdm_stream(), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                results.append(future.result())

    _touch_marker(out_dir, "DOWNLOAD_COMPLETE.txt", f"Dataset=eegmmidb\nURL={source_url}\nFiles={len(results)}\n")
    resolved_version = _infer_dataset_version_label(source_url, "1.0.0")
    write_json(
        out_dir / "DOWNLOAD_INDEX.json",
        {
            "dataset": "eegmmidb",
            "source_url": source_url,
            "dataset_version": resolved_version,
            "runs": runs,
            "subjects": subjects or "all",
            "file_count": len(results),
            "items": results,
        },
    )
    return DownloadRecord(
        dataset_name="eegmmidb",
        source_url=source_url,
        downloaded_at_utc=_now(),
        local_path=str(out_dir),
        status="skipped_existing" if results and all(item["status"] == "skipped_existing" for item in results) else "downloaded",
        version=resolved_version,
        notes=f"Downloaded EDF files for runs {runs} and {len(subjects or list(range(1, 110)))} subjects using direct parallel PhysioNet HTTP.",
        file_count=len(results),
        bytes_downloaded=sum(int(item.get("bytes", 0) or 0) for item in results),
    )


def _iter_ptbxl_targets(csv_path: Path, target_sampling_rate_hz: int, max_records: int | None) -> Iterable[str]:
    file_col = "filename_lr" if int(target_sampling_rate_hz) == 100 else "filename_hr"
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            stem = row[file_col]
            yield stem + ".hea"
            yield stem + ".dat"
            if max_records is not None and idx >= int(max_records):
                break


def _download_ptbxl(out_dir: Path, source_url: str, target_sampling_rate_hz: int, timeout: int, retries: int, workers: int, max_records: int | None) -> DownloadRecord:
    """Download PTB-XL metadata plus waveform pairs.

    Inputs:
    - out_dir: dataset raw directory.
    - source_url: PhysioNet base URL.
    - target_sampling_rate_hz: 100 or 500 Hz waveform tree.
    - timeout: HTTP timeout in seconds.
    - retries: number of attempts.
    - workers: parallel file worker count.
    - max_records: optional development subset cap.

    Outputs:
    - Download manifest record describing the PTB-XL acquisition.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = source_url.rstrip("/") + "/"
    resolved_version = _infer_dataset_version_label(source_url, "1.0.3")
    print(f"    {status_text('INFO', 'info')} Downloading PTB-XL metadata files into {out_dir}", flush=True)
    for meta_name in _METADATA_FILES:
        _stream_http(base_url + meta_name, out_dir / meta_name, timeout=timeout, retries=retries, show_progress=True)

    csv_path = out_dir / "ptbxl_database.csv"
    targets = sorted(set(_iter_ptbxl_targets(csv_path, target_sampling_rate_hz=target_sampling_rate_hz, max_records=max_records)))
    total = len(targets)
    print(f"    {status_text('INFO', 'info')} Downloading PTB-XL waveform files for {total // 2} records at {target_sampling_rate_hz} Hz with {workers} workers", flush=True)

    def _task(rel_path: str) -> str:
        dest = out_dir / rel_path
        _stream_http(
            base_url + rel_path,
            dest,
            timeout=timeout,
            retries=retries,
            show_progress=(max(1, workers) == 1),
        )
        return rel_path

    if max(1, workers) == 1:
        file_bar = tqdm(targets, total=total, desc="PTB-XL files", unit="file", dynamic_ncols=True, file=_tqdm_stream(), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        for rel in file_bar:
            file_bar.set_postfix_str(rel)
            _task(rel)
        file_bar.close()
    else:
        with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            future_map = {ex.submit(_task, rel): rel for rel in targets}
            for future in tqdm(cf.as_completed(future_map), total=total, desc="PTB-XL files", unit="file", dynamic_ncols=True, file=_tqdm_stream(), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
                future.result()

    _touch_marker(out_dir, "DOWNLOAD_COMPLETE.txt", f"Dataset=ptbxl\nURL={source_url}\nSamplingRate={target_sampling_rate_hz}\n")
    return DownloadRecord(
        dataset_name="ptbxl",
        source_url=source_url,
        downloaded_at_utc=_now(),
        local_path=str(out_dir),
        status="downloaded",
        version=resolved_version,
        notes=f"Downloaded metadata plus {'all available' if max_records is None else max_records} PTB-XL records at {target_sampling_rate_hz} Hz using streamed parallel HTTP.",
        file_count=2 + total,
        bytes_downloaded=sum((out_dir / rel).stat().st_size for rel in targets if (out_dir / rel).exists()) + sum((out_dir / meta).stat().st_size for meta in _METADATA_FILES if (out_dir / meta).exists()),
    )


def _simulate_record(name: str, url: str, out_dir: Path, notes: str, version: str | None = None) -> DownloadRecord:
    _touch_marker(out_dir, "DOWNLOAD_SIMULATED.txt", f"Dataset={name}\nURL={url}\n")
    return DownloadRecord(
        dataset_name=name,
        source_url=url,
        downloaded_at_utc=_now(),
        local_path=str(out_dir),
        status="simulated",
        version=version,
        notes=notes,
        file_count=1,
        bytes_downloaded=0,
    )


def _write_setup_summary(path: Path, manifest: list[dict], mode: str) -> None:
    lines = ["# Setup summary", "", f"Mode: {mode}", ""]
    failures = [m for m in manifest if m["status"] == "failed"]
    for m in manifest:
        lines.append(f"- {m['dataset_name']}: {m['status']} ({m.get('file_count', 'n/a')} files, {m.get('bytes_downloaded', 'n/a')} bytes)")
        if m.get("notes"):
            lines.append(f"  - notes: {m['notes']}")
    lines += ["", f"Overall status: {'FAILED' if failures and mode == 'strict' else 'PARTIAL' if failures else 'SUCCESS'}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_compliance_checklist(path: Path, manifest: list[dict], include_mhealth: bool, mode: str) -> None:
    """Write a machine-readable brief-compliance checklist for setup-stage artefacts."""
    statuses = {m["dataset_name"]: m["status"] for m in manifest}
    checklist = {
        "mode": mode,
        "include_mhealth": include_mhealth,
        "wrapper_created_required_directories": True,
        "download_manifest_written": True,
        "setup_summary_written": True,
        "datasets": {name: {"status": status, "present": status in {"downloaded", "simulated"}} for name, status in statuses.items()},
    }
    write_json(path, checklist)


def perform_setup(project_root: Path, include_mhealth: bool = True, simulate_downloads: bool = False, config_path: str | Path = "configs/default.yaml", mode_override: str | None = None, workers_override: int | None = None) -> list[dict]:
    cfg = _load_config(project_root, config_path)
    paths = setup_folders(project_root)
    resolved_config = project_root / config_path if not Path(config_path).is_absolute() else Path(config_path)
    update_reproducibility_context(paths["reports"], resolved_config, "setup")
    datasets = dict(cfg["datasets"])
    if not include_mhealth:
        datasets.pop("mhealth", None)

    runtime = cfg.get("runtime", {})
    mode = mode_override or runtime.get("setup_mode", "strict")
    timeout = int(runtime.get("http_timeout_seconds", 60))
    retries = int(runtime.get("retry_count", 3))
    workers = int(workers_override or runtime.get("setup_workers", 4))
    remove_archives_after_extract = bool(runtime.get("remove_archives_after_extract", True))

    progress = ProgressPrinter(total_steps=len(datasets) + 3)
    progress.stage("Creating repository folders for raw, interim, processed, reports, and submission samples")
    progress.stage(f"Preparing scripted dataset acquisition (mode={mode})")

    manifest: list[dict] = []
    failures: list[str] = []
    for i, (name, ds_cfg) in enumerate(datasets.items(), start=1):
        out_dir = ensure_dir(paths["raw"] / name)
        url = ds_cfg["source_url"]
        version = ds_cfg.get("dataset_version")
        required = bool(ds_cfg.get("required", True))
        progress.item(i, len(datasets), f"{name}: {url}")
        try:
            if simulate_downloads:
                record = _simulate_record(name, url, out_dir, notes="Simulated dataset acquisition for offline testing.", version=version)
            else:
                if name in {"pamap2", "wisdm", "mhealth"}:
                    record = _download_uci_dataset(
                        name,
                        url,
                        out_dir,
                        timeout=timeout,
                        retries=retries,
                        remove_archives_after_extract=remove_archives_after_extract,
                        dataset_version=version,
                    )
                elif name == "eegmmidb":
                    record = _download_eegmmidb(
                        out_dir,
                        url,
                        cfg["eeg"]["runs"],
                        cfg["eeg"].get("subjects"),
                        timeout=timeout,
                        retries=retries,
                        workers=int(cfg.get("eeg", {}).get("download_workers", workers)),
                    )
                elif name == "ptbxl":
                    record = _download_ptbxl(
                        out_dir,
                        url,
                        target_sampling_rate_hz=int(cfg["ecg"]["sampling_rate_hz"]),
                        timeout=timeout,
                        retries=retries,
                        workers=int(runtime.get("ptbxl_download_workers", workers)),
                        max_records=cfg.get("ecg", {}).get("max_records"),
                    )
                else:  # pragma: no cover
                    raise RuntimeError(f"Unhandled dataset: {name}")
        except Exception as exc:
            _touch_marker(out_dir, "DOWNLOAD_FAILED.txt", f"Dataset={name}\nURL={url}\nError={exc}\n")
            record = DownloadRecord(
                dataset_name=name,
                source_url=url,
                downloaded_at_utc=_now(),
                local_path=str(out_dir),
                status="failed",
                version=version,
                notes=str(exc),
                file_count=0,
            )
            if required:
                failures.append(name)
            print(f"    {status_text('ERROR', 'error')} dataset {name} failed: {exc}", flush=True)
            if mode == "strict":
                print(f"    {status_text('WARNING', 'warn')} strict mode: recording failure and continuing summary generation; setup will fail at the end.", flush=True)
        manifest.append(asdict(record))

    progress.stage("Writing machine-readable download manifest and summary")
    write_json(paths["reports"] / "download_manifest.json", manifest)
    _write_setup_summary(paths["reports"] / "setup_summary.md", manifest, mode=mode)
    _write_compliance_checklist(paths["reports"] / "compliance_checklist.json", manifest, include_mhealth=include_mhealth, mode=mode)

    progress.stage("Setup complete")
    if failures and mode == "strict":
        raise RuntimeError(f"Setup failed for required dataset(s): {', '.join(failures)}")
    return manifest


def main() -> int:
    args = parse_args()
    perform_setup(
        Path(args.project_root).resolve(),
        include_mhealth=args.include_mhealth,
        simulate_downloads=args.simulate_downloads,
        config_path=args.config,
        mode_override=args.mode,
        workers_override=args.workers,
    )
    print(f"{status_text('OK', 'ok')} Setup complete. Directory structure, raw folders, and setup reports are ready.", flush=True)
    return 0


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    with stage_log("setup", REPO_ROOT):
        print(f"{status_text('INFO', 'info')} [mmprep] setup_http_headers_log={_get_http_debug_path()}", flush=True)
        raise SystemExit(main())
