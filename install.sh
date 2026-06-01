#!/usr/bin/env bash
# install.sh — MAP144 install / update for Linux + macOS.
#
# Run from inside the cloned repo (no auto-clone — gives you full
# control of where the working tree lives):
#
#     git clone https://github.com/wa1hco/map144.git
#     cd map144
#     ./install.sh
#
# Re-runnable: existing venv is reused, requirements re-resolved.
# Sets up:
#   - .venv/ — Python virtual environment with pinned deps
#   - jt9 discovery verified
#   - run.sh ready to launch
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Find a working Python ≥ 3.10.  Prefer ``python3`` since that's the
# convention every modern distro uses.
PYTHON_CMD="${MAP144_PYTHON:-python3}"
if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
    echo "ERROR: '$PYTHON_CMD' not found." >&2
    echo "  Install Python 3.10+ from your distro package manager." >&2
    echo "  Or set MAP144_PYTHON to the full path of a python3 binary." >&2
    exit 1
fi

# Version check — bail early so we don't ship a confusing pip/wheel error.
"$PYTHON_CMD" -c '
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python {sys.version_info[:3]} found; MAP144 needs >= 3.10.")
print(f"Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]} OK")
'

# Create venv if missing.  Idempotent: re-runs reuse the existing one.
#
# --system-site-packages is REQUIRED for the USRP/B210 source: the UHD Python
# bindings ship only as a system package (apt python3-uhd, in
# /usr/lib/python3/dist-packages/uhd) and have no pip wheel.  A plain isolated
# venv cannot see them, so map144 reports "UHD Python package not found" even
# though it's installed.  Exposing system site-packages lets the venv import
# uhd while pip-installed deps (numpy<2 etc.) still shadow their system copies.
if [ ! -d "$REPO_DIR/.venv" ]; then
    echo "Creating venv at $REPO_DIR/.venv (with --system-site-packages for UHD) ..."
    "$PYTHON_CMD" -m venv --system-site-packages "$REPO_DIR/.venv"
elif ! grep -q "include-system-site-packages = true" "$REPO_DIR/.venv/pyvenv.cfg" 2>/dev/null; then
    # Existing venv predates this requirement — flip it on in place so UHD
    # becomes visible without a full rebuild.
    echo "Enabling system-site-packages on existing venv (for UHD) ..."
    sed -i 's/include-system-site-packages = false/include-system-site-packages = true/' \
        "$REPO_DIR/.venv/pyvenv.cfg"
fi

# Use the venv's python directly — avoids needing to source activate.
VENV_PY="$REPO_DIR/.venv/bin/python"

echo "Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip

echo "Installing requirements ..."
"$VENV_PY" -m pip install -r requirements.txt

# Verify jt9 discovery.  Don't hard-fail — the operator can install
# WSJT-X afterwards or set MAP144_JT9; surfacing a warning is enough.
echo "Verifying jt9 discovery ..."
JT9_PATH="$("$VENV_PY" -c 'from map144_app.detection import find_jt9; p = find_jt9(); print(p if p else "")')"
if [ -z "$JT9_PATH" ]; then
    echo "WARNING: jt9 not found.  Install WSJT-X (provides jt9), or set" >&2
    echo "         MAP144_JT9 to the full path of the jt9 binary." >&2
else
    echo "jt9: $JT9_PATH"
fi

# Resolve the installed version from the package itself — single source
# of truth, no risk of the install banner drifting from what the app
# actually reports.
MAP144_VERSION="$("$VENV_PY" -c 'from map144_app import __version__; print(__version__)' 2>/dev/null || echo '???')"

cat <<EOF

MAP144 v$MAP144_VERSION ready.

To run:
    ./run.sh

Or directly:
    $VENV_PY map144.py

The first launch may take ~10 s while numba JIT-compiles its hot paths.
EOF
