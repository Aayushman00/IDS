#!/usr/bin/env bash
# Scenario only (no IDS), writes a NetAnim trace. Short + low-rate UDP flood so the XML stays small.
#   bash ns3_sim/run_anim.sh            -> ns3_sim/results/anim/anim.xml
set -euo pipefail
cd "$(dirname "$0")/.."
NS3=$HOME/ns3/ns-allinone-3.43/ns-3.43
OUT="$PWD/ns3_sim/results/anim"; mkdir -p "$OUT"
cp ns3_sim/iot-ids.cc "$NS3/scratch/iot-ids.cc"
( cd "$NS3" && ./ns3 run "iot-ids --seed=1 --out=$OUT --duration=60 --udpKbps=8 --anim=$OUT/anim.xml $*" )
ls -lh "$OUT/anim.xml"
echo "open with: $NS3/../netanim-3.109/NetAnim  (then File > Open XML)"
