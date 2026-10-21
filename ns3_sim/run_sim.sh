#!/usr/bin/env bash
# ns-3 scenario -> pcap -> features -> CNN-LSTM metrics. From iot_ids_cnn_lstm/ in WSL:
#   tools/detach.sh sim.log 'bash ns3_sim/run_sim.sh 1'      # arg1 = seed (default 1); extra args go to the scenario
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-1}"; shift || true
NS3=$HOME/ns3/ns-allinone-3.43/ns-3.43
OUT="$PWD/ns3_sim/results/seed$SEED"; mkdir -p "$OUT"
cp ns3_sim/iot-ids.cc "$NS3/scratch/iot-ids.cc"
( cd "$NS3" && ./ns3 run "iot-ids --seed=$SEED --out=$OUT $*" )
PY=$HOME/venvs/ids/bin/python
$PY ns3_sim/extract_features.py "$OUT"/gw-*.pcap "$OUT/labels.csv" "$OUT/features.csv"
$PY ns3_sim/ids_infer.py "$OUT/features.csv" "$OUT/metrics.json"
