"""Legend + timeline card for the NetAnim view (NetAnim has no legend). Colours match iot-ids.cc.
usage: python ns3_sim/make_legend.py [duration=60]  -> ns3_sim/results/anim/legend.png"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

D = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
slot = D / 6
rgb = lambda r, g, b: (r / 255, g / 255, b / 255)
NODES = [("IoT device (benign sensor)", rgb(40, 200, 80)), ("Gateway + IDS tap", rgb(30, 100, 255)),
         ("Server", rgb(150, 60, 200)), ("Attacker, idle", rgb(150, 150, 150))]
ATK = [("UDP flood", rgb(255, 140, 0)), ("ICMP flood", rgb(235, 200, 0)),
       ("SYN flood", rgb(230, 30, 30)), ("Port scan", rgb(200, 40, 200))]

fig, (ax, tl) = plt.subplots(2, 1, figsize=(9, 5.2), gridspec_kw={"height_ratios": [3, 1.3]})
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
ax.text(0, 5.6, "How to read the NetAnim view", fontsize=15, weight="bold", va="center")
for i, (name, c) in enumerate(NODES):
    y = 4.6 - i * 0.8
    ax.scatter([0.3], [y], s=420, c=[c], edgecolors="black", linewidths=0.6)
    ax.text(0.8, y, name, va="center", fontsize=11)
ax.text(5.2, 4.6, "Attacker colour while active", fontsize=11, weight="bold", va="center")
for i, (name, c) in enumerate(ATK):
    y = 3.8 - i * 0.8
    ax.scatter([5.5], [y], s=420, c=[c], edgecolors="black", linewidths=0.6)
    ax.text(6.0, y, name, va="center", fontsize=11)
ax.text(0, 0.6, "Top: attackers grouped by type · bottom-left: IoT devices · centre-right: gateway (IDS) · far right: server.\n"
        "Moving dots = packets. Attackers start grey and light up only during their attack slot.", fontsize=9.5, va="center", color="#444")

tl.set_xlim(0, D); tl.set_ylim(0, 1); tl.set_yticks([])
for s in ("top", "right", "left"):
    tl.spines[s].set_visible(False)
segs = [("benign only", rgb(40, 200, 80)), *ATK, ("benign only", rgb(40, 200, 80))]
for i, (name, c) in enumerate(segs):
    tl.add_patch(Rectangle((i * slot, 0.15), slot, 0.7, fc=c, ec="white", lw=2, alpha=0.9))
    tl.text(i * slot + slot / 2, 0.5, name, ha="center", va="center", fontsize=8.5, weight="bold",
            color="white" if name != "ICMP flood" else "black")
tl.set_xlabel("simulation time (s)"); tl.set_title("Timeline (benign traffic runs the whole time)", fontsize=10, loc="left")
fig.tight_layout()
out = Path(__file__).parent / "results/anim/legend.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160)
print(out)
