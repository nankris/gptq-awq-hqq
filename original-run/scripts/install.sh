#!/usr/bin/env bash
# Run on the dedicated Colab VM. Each library gets at most 20 minutes.
set -u
mkdir -p results
timeout 1200 pip install hqq==0.2.8.post1 accelerate matplotlib > results/install_hqq.log 2>&1
timeout 1200 pip install --no-build-isolation gptqmodel==7.6.0 > results/install_gptq.log 2>&1
timeout 1200 pip install llmcompressor==0.14.0 > results/install_awq.log 2>&1
pip freeze > results/requirements.txt
