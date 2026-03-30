import subprocess
import sys
import json
from tests.conftest import make_fixture_repo


def test_root_entrypoints_smoke(tmp_path):
    root = make_fixture_repo(tmp_path)
    repo = __import__('pathlib').Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(repo / 'preprocess.py'), '--config', str(root / 'config.yaml')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, str(repo / 'validate_outputs.py'), '--processed-dir', str(root / 'data/processed'), '--manifest-path', str(root / 'reports/processed_manifest.json'), '--report-path', str(root / 'reports/validation_report.md'), '--sample-dir', str(root / 'submission_sample')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    context = json.loads((root / 'reports' / 'reproducibility_context.json').read_text(encoding='utf-8'))
    assert {'setup', 'preprocess', 'validate'}.issubset(set(context.get('completed_stages', [])))
