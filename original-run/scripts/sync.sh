#!/usr/bin/env bash
set -eu
[ -n "${TOOLS_SETUP:-}" ] && source "$TOOLS_SETUP"  # optional: environment setup providing the colab CLI
cd "$(dirname "$0")/.."
timeout 50 colab exec -s original-run < scripts/poll.py
timeout 50 colab download -s original-run /content/original-run/checkpoint.tar.gz results/checkpoint.tar.gz
tar -xzf results/checkpoint.tar.gz
