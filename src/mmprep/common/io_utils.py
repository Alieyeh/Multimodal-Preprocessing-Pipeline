"""Common file and configuration helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it.

    Inputs:
    - path: target directory path.

    Outputs:
    - Path object pointing to the created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Inputs:
    - path: YAML file path.

    Outputs:
    - Parsed mapping.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: str | Path, obj: Any) -> None:
    """Write JSON with deterministic formatting."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_array_npz(path: str | Path, array: np.ndarray) -> None:
    """Write a compressed array bundle with standard key `X`."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, X=array.astype(np.float32, copy=False))


def write_dataframe(path: str | Path, df: pd.DataFrame) -> None:
    """Write a CSV metadata table."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
