#!/usr/bin/env bash
# Send a short shell command to only this experiment's runtime.
set -eu
[ -n "${TOOLS_SETUP:-}" ] && source "$TOOLS_SETUP"  # optional: environment setup providing the colab CLI
cd "$(dirname "$0")/.."
{ sleep 5; printf '%s\n' "$@"; sleep 8; } | timeout 55 colab console -s quant-noninferiority > results/remote_latest.console.log 2>&1
"${PYTHON:-python3}" scripts/clean_console.py results/remote_latest.console.log
