#!/usr/bin/env python3
"""Root-level preprocessing entrypoint.

Expected inputs:
- Config file path.
- Raw data directories created by setup stage.
- Optional synthetic or test fixtures for dry runs.

Expected outputs:
- Interim artefacts under data/interim.
- Final fixed-shape processed outputs under data/processed.
- Machine-readable processed manifest in reports/processed_manifest.json.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmprep.cli import preprocess_main
from mmprep.common.logging_utils import stage_log

if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    with stage_log("preprocess", REPO_ROOT):
        raise SystemExit(preprocess_main())
