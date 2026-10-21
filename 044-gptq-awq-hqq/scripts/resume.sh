#!/usr/bin/env bash
set -eu
source /workspace/tools/setup.sh
cd /workspace/work/tig/044-gptq-awq-hqq
colab sessions
# Inspect before launching: detached jobs survive local reboots.
colab exec -s b6-044 -f scripts/poll.py
colab download -s b6-044 /content/exp044/checkpoint.tar.gz results/checkpoint.tar.gz
tar -xzf results/checkpoint.tar.gz
# If driver is dead and driver_done.json absent, inspect errors before restarting.
# Upload run.py / driver.py to /content/exp044 and start with:
# subprocess.Popen('cd /content/exp044 && nohup python -u driver.py > results/driver.log 2>&1 < /dev/null &',shell=True,start_new_session=True)
# Stop b6-044 only after all measurements and final checkpoint transfer.
