#!/usr/bin/env bash
set -eu
source /workspace/tools/setup.sh
cd /workspace/work/tig/044b-quant-noninferiority
# Jupyter channels fail on this runtime; its terminal websocket works.
{ sleep 5; printf '%s\n' 'python /content/exp044b/poll.py'; sleep 10; } | timeout 55 colab console -s b7-044b > results/poll_latest.console.log 2>&1
/workspace/tools/mlenv/bin/python scripts/clean_console.py results/poll_latest.console.log
timeout 55 colab download -s b7-044b /content/exp044b/checkpoint.tar.gz results/checkpoint.download.tar.gz
gzip -t results/checkpoint.download.tar.gz
mv results/checkpoint.download.tar.gz results/checkpoint.tar.gz
tar -xzf results/checkpoint.tar.gz
/workspace/tools/mlenv/bin/python -c 'import json,time; d=json.load(open("results/collection.json")); age=time.time()-d["collected_unix"]; assert -60<age<180, f"Stale checkpoint: {age:.1f} seconds"; print(f"Fresh checkpoint: {age:.1f} seconds old")'
/workspace/tools/mlenv/bin/python scripts/compare_completed.py
