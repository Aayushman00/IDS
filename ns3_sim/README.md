# ns-3 simulation of the CNN-LSTM IoT IDS

```
ns-3 scenario (iot-ids.cc) -> gateway pcap + labels.csv -> extract_features.py -> ids_infer.py -> metrics.json
```

Topology: benign IoT devices (10.1.1.0/24) and attackers (10.1.2.0/24) hang off a gateway; the gateway's
server-side port (10.1.3.2 = server) is tapped. The IDS lives at the gateway.
Timeline (6 equal slots): benign | UDP flood | ICMP flood | SYN flood | port scan | benign.

## Run (WSL, one-time setup then repeatable)

```bash
cd /mnt/c/Users/ayush/Desktop/Coding/IDS/iot_ids_cnn_lstm
tools/detach.sh ns3_setup.log 'bash ns3_sim/setup_ns3.sh'      # builds ns-3.43 in ~/ns3 (once)
tools/detach.sh sim.log 'bash ns3_sim/run_sim.sh 1'            # seed 1; extra args -> scenario
# e.g. bash ns3_sim/run_sim.sh 2 --nBenign=50 --nAtk=5 --duration=300
```

Results: `ns3_sim/results/seed<N>/{gw-*.pcap, labels.csv, features.csv, predictions.csv, metrics.json}`.
Check the extractor: `~/venvs/ids/bin/python -m pytest tests/test_ns3_extract.py`.

## Caveats (read before quoting numbers)

* Features are **approximations** of the CICIoT2023 extractor (not public in detail); see `extract_features.py`
  docstring. `metrics.json: out_of_range_share` = windows outside the training feature range, a domain-shift gauge.
* ns-3 attacks are behavioural stand-ins (no spoofed SYN, no Mirai bots, no HTTP/DNS/SSH apps): UDP/ICMP flood,
  TCP connect flood to a closed port (SYN+RST), sequential port scan.
* Label = per host and time slot (`labels.csv`); a window takes the label active at its first packet.
* Expect lower accuracy than the 98 % on real CICIoT2023; report it as a sim-to-real generalisation result.

## Findings (seeds 1-5, 20 benign devices, 3 attackers/type, 120 s)

**Calibration against 166k real CICIoT2023 rows** (`extract_features.py` / `iot-ids.cc` comments list each rule):
real `Drate==0` and `Srate==Rate`; `Magnitue==sqrt(2*AVG)`, `Radius==sqrt(2)*Std` (exact); real attack frames are
tiny (50/42/54 B) and low-rate. After calibration only 3 % of sim windows fall outside the training range (was 61 %).

**Zero-shot (real model on sim): fails, all 5 seeds** (accuracy 4-6 %, ROC-AUC 0.06-0.18, UDP flood 0 % detected).
Cause, verified by `ablate_real.py` on the REAL test set: neutralising `IAT` alone drops benign recall 99.9 % -> 2.8 %.
The model is essentially an `IAT` classifier, and CICIoT2023's `IAT` is a capture artifact (~0.04 s for benign, multiples
of ~8.3e7 for attacks; `Number`, `Weight` are likewise generator constants). ns-3 cannot reproduce this honestly.
So the 98 % on real data does not transfer to a physically meaningful traffic stream. Worth a paragraph in the report.

**Fine-tuned on sim seeds 1-2 (val 3), tested on held-out seeds 4-5** (`finetune_sim.py`, sample weights equalise labels):
all 4 attack types 100 % detected, benign recall 79 %, macro-recall 0.958, ROC-AUC 0.997 (zero-shot macro-recall 0.635).
Caveat: same topology/generator for train and test, so this is sim-to-sim, not sim-to-real.
Raw window-count accuracy flatters (UDP flood = ~96 % of attack windows); use macro_recall / per-label rates.

**Retrained without `IAT`, `Number`, `Weight`** (`python main.py --stage prepare,train --sample 1400000 --epochs 25
--output_dir outputs_noartifact --drop_features IAT Number Weight`; same data/split/seed as the main run):
* Real test set barely changes: accuracy 98.45 % (was 98.74 %), benign recall 99.63 % (99.84 %), MCC 0.787 (0.818).
  So the headline 98 % does not need those three columns; other features carry the same shortcut.
* On sim seeds 1-5 (zero-shot, `ids_infer.py ... outputs_noartifact`): no longer inverted but still useless. The model flags
  EVERY window as malicious (benign included): accuracy 91-95 % only because ~90 % of sim windows are attacks, ROC-AUC
  0.49-0.75 (chance level), macro-recall 0.80 on all 5 seeds. Per-seed files: `results/seed*/metrics_noartifact.json`.
* Reading: removing the three artifacts is not enough; the real-trained model separates CICIoT2023's capture style, not
  physical traffic behaviour. Fine-tuning on sim data (above) is what makes the detector work on ns-3 traffic.

## Visualisation
`bash ns3_sim/run_anim.sh` -> `results/anim/anim.xml` (open in NetAnim); `python ns3_sim/make_legend.py` -> `results/anim/legend.png`
(colour legend + attack timeline for slides). Attackers turn orange/yellow/red/magenta only during their slot.
