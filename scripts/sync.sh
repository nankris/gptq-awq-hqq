#!/usr/bin/env bash
set -eu
source /workspace/tools/setup.sh
cd /workspace/work/tig/044-gptq-awq-hqq
timeout 50 colab exec -s b6-044 < scripts/poll.py
timeout 50 colab download -s b6-044 /content/exp044/checkpoint.tar.gz results/checkpoint.tar.gz
tar -xzf results/checkpoint.tar.gz
