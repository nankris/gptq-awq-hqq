#!/usr/bin/env bash
set -eu
[ -n "${TOOLS_SETUP:-}" ] && source "$TOOLS_SETUP"  # optional: environment setup providing the colab CLI
cd "$(dirname "$0")/.."
colab sessions
# Inspect before launching: detached jobs survive local reboots.
colab exec -s original-run -f scripts/poll.py
colab download -s original-run /content/original-run/checkpoint.tar.gz results/checkpoint.tar.gz
tar -xzf results/checkpoint.tar.gz
# If driver is dead and driver_done.json absent, inspect errors before restarting.
# Upload run.py / driver.py to /content/original-run and start with:
# subprocess.Popen('cd /content/original-run && nohup python -u driver.py > results/driver.log 2>&1 < /dev/null &',shell=True,start_new_session=True)
# Stop original-run only after all measurements and final checkpoint transfer.
