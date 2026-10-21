"""Seed's simulated windows + the 3 models' verdicts -> ns3_sim/demo/sim_data.js (window.SIM) for demo/index.html.
Per host and simulated second we store 8 ints: [n_attack, n_benign, then (flagged_attack, flagged_benign) per model].
usage: python ns3_sim/export_demo.py [seed=4]   (from iot_ids_cnn_lstm/, WSL venv; seed 4 = held out from fine-tuning)"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
d = pd.read_csv(HERE / f"results/seed{seed}/features.csv")
labels = list(csv.DictReader(open(HERE / f"results/seed{seed}/labels.csv")))
ft = json.load(open(HERE / "results/finetune.json"))["setup"]

MODELS = [  # (title, short, outputs dir for scaler+features, keras file)
    ("Real-trained (CICIoT2023)", "A", "outputs", ROOT / "outputs/models/CNN-LSTM.keras"),
    ("Retrained without IAT / Number / Weight", "B", "outputs_noartifact", ROOT / "outputs_noartifact/models/CNN-LSTM.keras"),
    ("Fine-tuned on ns-3 (seeds %s)" % "+".join(map(str, ft["train"][:-1])), "C", "outputs", HERE / "results/CNN-LSTM_sim_ft.keras"),
]
flag = []
for title, _, od, path in MODELS:
    cols = json.load(open(ROOT / od / "cache/meta.json"))["feature_names"]
    X = joblib.load(ROOT / od / "models/scaler.joblib").transform(d[cols].to_numpy(np.float32))[..., None]
    flag.append(tf.keras.models.load_model(path).predict(X, batch_size=2048, verbose=0)[:, 1] >= 0.5)

hosts = [{"ip": r["ip"], "label": r["label"]} for r in labels]
hid = {h["ip"]: i for i, h in enumerate(hosts)}
T = int(np.ceil(d.t.max())) + 1
cells = np.zeros((T, len(hosts), 2 + 2 * len(MODELS)), int)
atk = (d.label != "Benign").to_numpy()
for k, (ip, t) in enumerate(zip(d.host, d.t.to_numpy())):
    c = cells[int(t), hid[ip]]
    c[0 if atk[k] else 1] += 1
    for m, f in enumerate(flag):
        c[2 + 2 * m + (0 if atk[k] else 1)] += int(f[k])
out = {"seed": seed, "T": T, "hosts": hosts, "models": [m[0] for m in MODELS], "cells": cells.tolist(),
       "window_packets": 10,
       "totals": {m[1]: {"tp": int(f[atk].sum()), "fn": int((~f[atk]).sum()), "fp": int(f[~atk].sum()),
                         "tn": int((~f[~atk]).sum())} for m, f in zip(MODELS, flag)},
       "by_label": {m[1]: {k: float(g.mean()) for k, g in pd.Series(f).groupby(d.label)} for m, f in zip(MODELS, flag)}}
(HERE / "demo").mkdir(exist_ok=True)
(HERE / "demo/sim_data.js").write_text("window.SIM=" + json.dumps(out, separators=(",", ":")) + ";\n")
print(json.dumps({"T": T, "hosts": len(hosts), "totals": out["totals"]}, indent=1))
