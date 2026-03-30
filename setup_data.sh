#!/usr/bin/env bash
set -euo pipefail
PYEXE="python"
if [ -x ".venv/bin/python" ]; then
  PYEXE=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYEXE="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYEXE="$(command -v python3)"
else
  echo "No python interpreter found. Install python/python3 or activate a virtual environment." >&2
  exit 127
fi

has_config=false
for arg in "$@"; do
  if [ "$arg" = "--config" ]; then
    has_config=true
    break
  fi
done

if [ "$has_config" = false ]; then
  exec "$PYEXE" setup_data.py --config configs/default.yaml "$@"
else
  exec "$PYEXE" setup_data.py "$@"
fi
