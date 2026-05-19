#!/usr/bin/env bash
# run.sh — launch MAP144 from its venv (Linux / macOS).
# Re-runnable; passes any args straight through to map144.py.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: .venv not found.  Run ./install.sh first." >&2
    exit 1
fi
exec .venv/bin/python map144.py "$@"
