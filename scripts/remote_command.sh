#!/usr/bin/env bash
# Send a short shell command to only this experiment's runtime.
set -eu
source /workspace/tools/setup.sh
cd /workspace/work/tig/044b-quant-noninferiority
{ sleep 5; printf '%s\n' "$@"; sleep 8; } | timeout 55 colab console -s b7-044b > results/remote_latest.console.log 2>&1
/workspace/tools/mlenv/bin/python scripts/clean_console.py results/remote_latest.console.log
