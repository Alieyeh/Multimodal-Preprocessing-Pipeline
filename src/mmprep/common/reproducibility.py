"""Helpers for recording lightweight reproducibility provenance across stages."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path


def _infer_completed_stages(reports_dir: Path) -> set[str]:
    """Infer completed stages from artefacts already present on disk."""
    completed: set[str] = set()
    if (reports_dir / "download_manifest.json").exists() and (reports_dir / "setup_summary.md").exists():
        completed.add("setup")
    if (reports_dir / "processed_manifest.json").exists() and (reports_dir / "preprocess_summary.md").exists():
        completed.add("preprocess")
    if (reports_dir / "validation_report.md").exists():
        completed.add("validate")
    for stage in ["setup", "preprocess", "validate"]:
        if (reports_dir / f"{stage}_metrics.json").exists():
            completed.add(stage)
    return completed


def update_reproducibility_context(reports_dir: str | Path, config_path: str | Path | None, stage_name: str) -> dict:
    """Append one completed stage to the reproducibility context and persist it."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    context_path = reports_dir / "reproducibility_context.json"
    payload: dict = {}
    if context_path.exists():
        try:
            payload = json.loads(context_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    prior_path = payload.get("config_path")
    path = Path(prior_path) if prior_path else Path()
    if config_path is not None:
        candidate = Path(config_path)
        if candidate.exists():
            path = candidate.resolve()
        elif not path.exists():
            path = candidate

    config_bytes = path.read_bytes() if path.exists() else b""
    completed = set(payload.get("completed_stages", []))
    completed.update(_infer_completed_stages(reports_dir))
    completed.add(stage_name)
    payload.update(
        {
            "config_path": str(path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest() if config_bytes else None,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "completed_stages": sorted(completed),
        }
    )
    context_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
