# How to present the ns-3 simulation to your professor

**One-sentence pitch:** *"We reproduced a 98 % CNN-LSTM IoT intrusion detector, then stress-tested it on fresh traffic from an ns-3
simulated IoT network. It failed in an informative way, and we found out why."*

Honest negative results with a diagnosed cause are worth more than a clean 98 % that nobody questioned.

## Storyline (about 8 minutes)

| # | Time | Say | Show |
|---|---|---|---|
| 1 | 1 min | Goal: the paper's model gets 98.4 % on CICIoT2023. Does it work on traffic it has never seen? | Your existing results slide (98.74 % on our reproduction) |
| 2 | 1 min | Why simulate: ground truth is known exactly, attacks are repeatable, we control the network. Tool: ns-3 on WSL. | `legend.png` or NetAnim topology |
| 3 | 1 min | The setup: 20 benign IoT sensors + 12 attackers (UDP flood, ICMP flood, SYN flood, port scan) -> gateway (IDS tap) -> server. 6 time slots. | NetAnim run (`results/anim/anim.xml`) |
| 4 | 1 min | Pipeline: pcap at gateway -> 10-packet windows per host -> the same 41 features -> model. Features calibrated against 166k real rows. | Diagram or `README.md` pipeline line |
| 5 | 3 min | **Live demo** (below). | `demo/index.html` |
| 6 | 1 min | Why it failed and what fixed it; limits; next steps. | Results table below |

## Live demo script (`ns3_sim/demo/index.html`, just double-click it)

1. Pick model **A (real-trained)**, press Play at 4x. Say: *"Green dots are normal sensors. Watch the UDP flood: the IDS says nothing."*
   Point to **Attacks missed 23,791** and the amber rings on benign devices (false alarms).
2. Pick **B (retrained without IAT / Number / Weight)**. Say: *"We removed the three columns we suspected were dataset artifacts.
   Real-data accuracy stayed 98.45 %, but now it flags everything."* (Benign passed stays 0.)
3. Pick **C (fine-tuned on ns-3 seeds 1+2)**. Say: *"Same network, but seed 4, which it never saw."*
   Attacks caught 24,640, missed 0, red rings appear only on attackers during their slot.
4. Drag the timeline slider to the SYN flood and port-scan slots to show every type is caught. Spacebar pauses.

Tip: pause on the UDP-flood slot with model A, then switch to C without moving the slider: same moment, opposite verdict.

## Numbers to quote (held-out seed 4, 26k windows)

| Model | Real CICIoT2023 test | Sim: attacks caught | Sim: benign passed | Verdict |
|---|---|---|---|---|
| A real-trained | 98.74 % acc | 3.4 % | 19 % | misses UDP flood, false alarms |
| B no IAT/Number/Weight | 98.45 % acc | 100 % | 0 % | flags everything |
| C fine-tuned on sim | n/a | 100 % | 78 % | works (macro-recall 0.958 on seeds 4+5) |

Why it fails (say this clearly): CICIoT2023's `IAT` column is a capture artifact (about 0.04 for benign, multiples of 8.3e7 for attacks).
Neutralising it alone drops real benign recall from 99.9 % to 2.8 %. A physical network cannot produce that value.
Removing three columns is not enough because other features carry similar dataset-specific shortcuts.

## Questions to expect, and answers

- **"Isn't the simulation just unrealistic?"** Partly, by design: attacks are behavioural stand-ins (no spoofing or botnet code).
  We calibrated the extractor to real rows (e.g. `Magnitue = sqrt(2*AVG)` exactly) and only 3 % of windows fall outside the training range.
  The model still fails, so the problem is the shortcut, not a few out-of-range values.
- **"Fine-tuning on the same simulator is circular."** Yes: it is sim-to-sim (different seeds, same topology). We say so. It shows the
  pipeline and features are sound, not that the model generalises to the real world. Next test: change topology or device count between train and test.
- **"Why 10-packet windows?"** CICIoT2023 computes features over 10-packet windows; we follow that.
- **"Why is accuracy 4 % for model A?"** About 90 % of simulated windows are attacks (a flood is thousands of packets per second), and A calls the biggest
  one (UDP flood) benign. Use macro-recall or per-attack rates, not accuracy.
- **"Does this invalidate the paper?"** No. The paper's number is valid on its dataset; we show it does not transfer to other traffic, which is a known
  weakness of single-dataset IDS evaluations.
- **"What would you do next?"** Vary topology/device mix; replace dataset-artifact features with physically measurable ones; add a second dataset.

## Do this the day before

```bash
# WSL: regenerate everything (about 2 minutes)
cd /mnt/c/Users/ayush/Desktop/Coding/IDS/iot_ids_cnn_lstm
for s in 1 2 3 4 5; do bash ns3_sim/run_sim.sh $s; done           # simulate + extract + evaluate
~/venvs/ids/bin/python ns3_sim/finetune_sim.py                     # model C
~/venvs/ids/bin/python ns3_sim/export_demo.py 4                    # refresh demo data
bash ns3_sim/run_anim.sh                                           # NetAnim trace
```

- Open `demo/index.html` in Chrome and test it once offline (no internet needed).
- Open NetAnim from WSL (`~/ns3/ns-allinone-3.43/netanim-3.109/NetAnim`) and load `anim.xml`; keep it open in a second window.
- Record a screen capture of the demo as a backup in case something breaks live.

## Slide skeleton (6 slides)

1. Problem and claim (98 % on CICIoT2023 reproduced)
2. Simulation design (topology picture + timeline, from `legend.png`)
3. Pipeline (ns-3 -> pcap -> features -> IDS), with the calibration check
4. Live demo
5. Results table (A / B / C) + the `IAT` ablation result
6. Limitations and next steps
