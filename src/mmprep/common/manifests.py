"""Manifest helpers."""
from __future__ import annotations

from pathlib import Path


def file_manifest_entry(path: str | Path, extra: dict | None = None) -> dict:
    """Build one machine-readable manifest record.

    Inputs:
    - path: path to an output file.
    - extra: optional extra metadata.

    Outputs:
    - Dictionary with file path and file size plus any extra fields.
    """
    p = Path(path)
    payload = {"path": str(p), "bytes": p.stat().st_size if p.exists() else None}
    if extra:
        payload.update(extra)
    return payload
