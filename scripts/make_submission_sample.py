#!/usr/bin/env python3
"""Create a brief-compliant representative processed sample pack.

Expected inputs:
- Processed `.npz` and `.csv` files in `data/processed`.
- Maximum total number of samples per source dataset.

Expected outputs:
- Sample files grouped under one folder per source dataset.
- A machine-readable sample-pack manifest describing expected counts and shapes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Bundle:
    dataset_name: str
    modality: str
    stem: str
    npz_path: Path
    csv_path: Path
    data: np.ndarray
    metadata: pd.DataFrame


def _select_indices(df: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Choose representative rows with light label balancing when available."""
    if n <= 0:
        return np.array([], dtype=int)
    if len(df) <= n:
        return np.arange(len(df))
    rng = np.random.default_rng(seed)
    if "label_or_event" not in df.columns:
        return np.sort(rng.choice(len(df), size=n, replace=False))
    groups: list[int] = []
    remaining = n
    labels = df["label_or_event"].fillna("<NA>").astype(str)
    unique = list(labels.unique())
    per_group = max(1, n // max(1, len(unique)))
    for label in unique:
        idx = df.index[labels == label].to_numpy()
        take = min(len(idx), per_group, remaining)
        if take > 0:
            groups.extend(rng.choice(idx, size=take, replace=False).tolist())
            remaining -= take
    if remaining > 0:
        selected = np.array(groups, dtype=int) if groups else np.array([], dtype=int)
        pool = np.setdiff1d(df.index.to_numpy(), selected, assume_unique=False)
        if len(pool) > 0:
            groups.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return np.sort(np.array(groups[:n], dtype=int))


def _discover_bundles(processed_dir: Path) -> list[Bundle]:
    """Load processed bundles with their metadata."""
    bundles: list[Bundle] = []
    for npz_path in sorted(processed_dir.glob("**/*.npz")):
        csv_path = npz_path.with_suffix(".csv")
        if not csv_path.exists():
            continue
        metadata = pd.read_csv(csv_path)
        if metadata.empty:
            continue
        bundles.append(
            Bundle(
                dataset_name=str(metadata["dataset_name"].iloc[0]),
                modality=str(metadata["modality"].iloc[0]),
                stem=npz_path.stem,
                npz_path=npz_path,
                csv_path=csv_path,
                data=np.load(npz_path, allow_pickle=False)["X"],
                metadata=metadata,
            )
        )
    return bundles


def _allocate_counts(bundles: list[Bundle], target_total: int) -> list[int]:
    """Allocate one dataset-level sample budget across its processed bundles."""
    capacities = [min(len(bundle.metadata), int(bundle.data.shape[0])) for bundle in bundles]
    total_capacity = sum(capacities)
    target = min(target_total, total_capacity)
    if target <= 0:
        return [0 for _ in bundles]

    counts = [0 for _ in bundles]
    active = [i for i, cap in enumerate(capacities) if cap > 0]
    # Ensure each non-empty bundle is represented before distributing the remainder.
    while active and sum(counts) < target:
        progressed = False
        for i in active:
            if counts[i] < capacities[i] and sum(counts) < target:
                counts[i] += 1
                progressed = True
        if not progressed:
            break

    remaining = target - sum(counts)
    while remaining > 0:
        spare = [capacities[i] - counts[i] for i in range(len(bundles))]
        total_spare = sum(max(0, value) for value in spare)
        if total_spare <= 0:
            break
        progressed = False
        for i, extra_capacity in enumerate(spare):
            if extra_capacity <= 0 or remaining <= 0:
                continue
            share = max(1, int(round((extra_capacity / total_spare) * remaining)))
            add = min(extra_capacity, share, remaining)
            if add > 0:
                counts[i] += add
                remaining -= add
                progressed = True
        if not progressed:
            break
    return counts


def main() -> int:
    """Generate a representative sample pack grouped by source dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="submission_sample")
    parser.add_argument("--n", type=int, default=100, help="Target total sample rows per source dataset.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing sample artefacts and the prior sample-pack manifest before writing the new pack.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for path in list(out_dir.glob("**/*_sample.csv")) + list(out_dir.glob("**/*_sample.npz")) + list(out_dir.glob("**/sample_summary.json")):
            path.unlink(missing_ok=True)
        (out_dir / "sample_pack_manifest.json").unlink(missing_ok=True)

    bundles = _discover_bundles(processed_dir)
    grouped: dict[str, list[Bundle]] = {}
    for bundle in bundles:
        grouped.setdefault(bundle.dataset_name, []).append(bundle)

    manifest_rows: list[dict] = []
    for dataset_name, dataset_bundles in sorted(grouped.items()):
        counts = _allocate_counts(dataset_bundles, args.n)
        target_total = min(args.n, sum(min(len(bundle.metadata), int(bundle.data.shape[0])) for bundle in dataset_bundles))
        written_total = 0
        dataset_summaries: list[dict] = []
        for offset, (bundle, count) in enumerate(zip(dataset_bundles, counts)):
            if count <= 0:
                continue
            idx = _select_indices(bundle.metadata, count, args.seed + offset)
            rel_dir = Path(bundle.modality) / dataset_name
            target_dir = out_dir / rel_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            sample_npz = target_dir / f"{bundle.stem}_sample.npz"
            sample_csv = target_dir / f"{bundle.stem}_sample.csv"
            sampled_array = bundle.data[idx].astype(np.float32, copy=False)
            sampled_metadata = bundle.metadata.iloc[idx].reset_index(drop=True)
            np.savez_compressed(sample_npz, X=sampled_array)
            sampled_metadata.to_csv(sample_csv, index=False)
            expected_shape = [int(count), int(bundle.data.shape[1]), int(bundle.data.shape[2])]
            written_total += int(count)
            manifest_rows.append(
                {
                    "dataset_name": dataset_name,
                    "modality": bundle.modality,
                    "source_stem": bundle.stem,
                    "source_npz": str(bundle.npz_path),
                    "source_csv": str(bundle.csv_path),
                    "sample_npz": str(sample_npz),
                    "sample_csv": str(sample_csv),
                    "source_rows": int(min(len(bundle.metadata), int(bundle.data.shape[0]))),
                    "expected_rows": int(count),
                    "expected_shape": expected_shape,
                    "dataset_target_rows": int(target_total),
                }
            )
            dataset_summaries.append(
                {
                    "source_stem": bundle.stem,
                    "modality": bundle.modality,
                    "sample_rows": int(count),
                    "sample_shape": expected_shape,
                    "sample_npz": str(sample_npz),
                    "sample_csv": str(sample_csv),
                    "window_kind": str(sampled_metadata["window_kind"].iloc[0]) if "window_kind" in sampled_metadata.columns and not sampled_metadata.empty else None,
                    "label_mode": (
                        "unlabeled"
                        if sampled_metadata["label_or_event"].fillna("").astype(str).str.strip().eq("").all()
                        else "labeled"
                    ),
                }
            )
        if written_total != target_total:
            raise RuntimeError(f"Sample allocation mismatch for {dataset_name}: wrote {written_total}, expected {target_total}")
        dataset_dir = out_dir / dataset_bundles[0].modality / dataset_name
        dataset_summary = {
            "dataset_name": dataset_name,
            "dataset_target_rows": int(target_total),
            "sample_rows_written": int(written_total),
            "files": dataset_summaries,
            "note": (
                "HAR datasets keep separate pretraining and supervised sample files so both required output types "
                "remain directly inspectable while the dataset-level sample total stays fixed."
                if dataset_bundles[0].modality == "har"
                else "This dataset has one processed output type, so the representative sample is contained in one file pair."
            ),
        }
        (dataset_dir / "sample_summary.json").write_text(json.dumps(dataset_summary, indent=2, sort_keys=True), encoding="utf-8")

    manifest = {
        "target_rows_per_dataset": int(args.n),
        "datasets": sorted(grouped),
        "files": manifest_rows,
    }
    (out_dir / "sample_pack_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
