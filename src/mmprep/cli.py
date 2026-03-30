from __future__ import annotations

import argparse
from pathlib import Path

from mmprep.common.console import status_text
from mmprep.common.reproducibility import update_reproducibility_context
from mmprep.pipeline import run_preprocess
from mmprep.validation import validate_processed

def preprocess_main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments and run preprocessing.

    Inputs:
    - argv: optional command-line argument list for testing.

    Outputs:
    - Process exit code from the preprocessing pipeline.
    """
    parser = argparse.ArgumentParser(description="Run multimodal preprocessing pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true", help="Skip processed outputs that already exist and continue from the next incomplete stage.")
    args = parser.parse_args(argv)
    return run_preprocess(args.config, resume=args.resume)


def validate_main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, validate processed outputs, and write the markdown report.

    Inputs:
    - argv: optional command-line argument list for testing.

    Outputs:
    - Zero on validation pass, otherwise one.
    """
    parser = argparse.ArgumentParser(description="Validate processed outputs.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--manifest-path", default="reports/processed_manifest.json")
    parser.add_argument("--report-path", default="reports/validation_report.md")
    parser.add_argument("--sample-dir", default="submission_sample")
    parser.add_argument("--config", default=None, help="Optional config path used for config-aware validation checks.")
    args = parser.parse_args(argv)

    print(f"{status_text('INFO', 'info')} Starting validation of processed outputs...", flush=True)
    report_path = Path(args.report_path)
    update_reproducibility_context(report_path.parent, args.config, "validate")
    ok, report = validate_processed(args.processed_dir, args.manifest_path, args.sample_dir, config_path=args.config)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    npz_count = len(list(Path(args.processed_dir).glob("**/*.npz"))) if Path(args.processed_dir).exists() else 0
    sample_csv_count = len(list(Path(args.sample_dir).glob("**/*.csv"))) if Path(args.sample_dir).exists() else 0
    print(f"{status_text('INFO', 'info')} Validation report written to: {report_path}", flush=True)
    print(f"{status_text('INFO', 'info')} Processed array bundles checked: {npz_count}", flush=True)
    print(f"{status_text('INFO', 'info')} Submission sample metadata files checked: {sample_csv_count}", flush=True)
    print(f"{status_text('PASS' if ok else 'FAIL', 'ok' if ok else 'error')} Validation status", flush=True)
    return 0 if ok else 1
