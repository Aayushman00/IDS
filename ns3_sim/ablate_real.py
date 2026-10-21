"""Which features does the trained CNN-LSTM lean on? Neutralise feature groups (set to the train mean)
on the REAL held-out test set and watch accuracy / attack recall. Big drop = reliance; for dataset-artifact
features (IAT, Number, Weight, Header_Length) that predicts failure on simulated traffic.
usage: python ns3_sim/ablate_real.py   (from iot_ids_cnn_lstm/, WSL venv)"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

C = Path(__file__).resolve().parent.parent / "outputs"
names = json.load(open(C / "cache/meta.json"))["feature_names"]
X, y = np.load(C / "cache/X_test.npy"), np.load(C / "cache/y_test.npy")
rng = np.random.default_rng(0)
i = rng.choice(len(X), 60000, replace=False)
X, y = X[i], y[i]
model = tf.keras.models.load_model(C / "models/CNN-LSTM.keras")
mean = np.load(C / "cache/X_train.npy", mmap_mode="r")[:200000].mean(0)


def score(Xa):
    yh = model.predict(Xa[..., None], batch_size=2048, verbose=0)[:, 1] >= 0.5
    return (yh == y).mean(), yh[y == 1].mean(), (~yh[y == 0]).mean()


GROUPS = {"none": [], "IAT": ["IAT"], "Number": ["Number"], "Weight": ["Weight"], "Header_Length": ["Header_Length"],
          "IAT+Number+Weight": ["IAT", "Number", "Weight"],
          "IAT+Number+Weight+Header_Length": ["IAT", "Number", "Weight", "Header_Length"],
          "Rate+Srate": ["Rate", "Srate"], "size feats": ["Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "Magnitue", "Radius"]}
print(f"{'ablated group':36s} acc    attack_recall  benign_recall")
res = {}
for g, cols in GROUPS.items():
    Xa = X.copy()
    for c in cols:
        Xa[:, names.index(c)] = mean[names.index(c)]
    res[g] = score(Xa)
    print(f"{g:36s} {res[g][0]:.4f} {res[g][1]:.4f}         {res[g][2]:.4f}")
json.dump(res, open(Path(__file__).parent / "results/ablation_real.json", "w"), indent=2)
