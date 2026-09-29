#!/usr/bin/env bash
# Run a pipeline command fully detached from the calling terminal / wsl.exe
# session (new session via `setsid -f`, stdin closed, output appended to a log).
# A killed terminal, IDE or agent session therefore cannot take the job down.
#
# usage:  tools/detach.sh <logfile> '<shell command>'
# e.g.    tools/detach.sh run_main.log 'python main.py --stage prepare,train --sample 1400000'
# `python` is mapped to the project venv. Poll with: tail -f <logfile>
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="$1"; CMD="$2"
PY="${IDS_PYTHON:-$HOME/venvs/ids/bin/python}"
CMD="${CMD//python /$PY -u }"
echo "===== $(date '+%F %T') launching: $CMD" >> "$LOG"
setsid -f bash -c "export TF_CPP_MIN_LOG_LEVEL=2; ( $CMD ) >> '$LOG' 2>&1; echo \"===== \$(date '+%F %T') exit=\$?\" >> '$LOG'" \
  < /dev/null > /dev/null 2>&1
sleep 1
echo "detached; log: $(pwd)/$LOG"
pgrep -af "main.py" | grep -v pgrep || true
