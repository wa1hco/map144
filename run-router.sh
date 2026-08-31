#!/usr/bin/env bash
# run-router.sh — launch the B210 audio/IQ router from the project venv.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
    echo "ERROR: .venv not found.  Run ./install.sh first." >&2
    exit 1
fi
exec .venv/bin/python router_app.py "$@"
