#!/usr/bin/env python3
"""Root-level validation entrypoint.

Expected inputs:
- Processed outputs in data/processed.
- Optional sample pack in submission_sample.
- Optional config path for config-aware validation checks.

Expected outputs:
- Validation report in reports/validation_report.md.
- Non-zero exit code if validation fails.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmprep.cli import validate_main
from mmprep.common.logging_utils import stage_log

if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    with stage_log("validate", REPO_ROOT):
        raise SystemExit(validate_main())
