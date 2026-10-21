#!/usr/bin/env bash
set -eu
mkdir -p /content/exp044b
cd /content/exp044b
tar -xzf /content/bundle044b.tar.gz
cp scripts/*.py scripts/install.sh .
mkdir -p results
nvidia-smi > results/nvidia_smi.txt
date -u +%FT%TZ > results/bootstrap_start.txt
bash install.sh
exec python -u driver.py
