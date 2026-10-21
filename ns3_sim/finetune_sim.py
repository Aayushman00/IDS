"""Zero-shot vs fine-tuned CNN-LSTM on simulated traffic, split BY SEED (train seeds never appear in test).
Starts from the real-data model, keeps the real scaler, fine-tunes with a small LR and balanced class weights.
Saved to ns3_sim/results/CNN-LSTM_sim_ft.keras (the real model in outputs/ is never touched).
usage: python ns3_sim/finetune_sim.py --train 1 2 3 --test 4 5   (from iot_ids_cnn_lstm/, WSL venv)"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
cols = json.load(open(OUT / "cache/meta.json"))["feature_names"]
scaler = joblib.load(OUT / "models/scaler.joblib")


def load(seeds):
    d = pd.concat([pd.read_csv(HERE / f"results/seed{s}/features.csv") for s in seeds], ignore_index=True)
    return scaler.transform(d[cols].to_numpy(np.float32))[..., None], (d.label != "Benign").to_numpy().astype(int), d.label


def report(model, X, y, lab):
    p = model.predict(X, batch_size=2048, verbose=0)[:, 1]
    yh = p >= 0.5
    return dict(accuracy=float((yh == y).mean()), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                benign_recall=float((~yh[y == 0]).mean()), roc_auc=roc_auc_score(y, p),
                detected_share_by_label=(by := {k: float(g.mean()) for k, g in pd.Series(yh).groupby(lab)}),
                # per-label "correct" rate (benign: not flagged); high-rate floods dominate raw window counts, so
                # accuracy flatters; macro_recall weights every label equally
                macro_recall=float(np.mean([1 - v if k == "Benign" else v for k, v in by.items()])))


ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--test", type=int, nargs="+", default=[4, 5])
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--lr", type=float, default=3e-4)
a = ap.parse_args()
tf.keras.utils.set_random_seed(42)
Xtr, ytr, ltr = load(a.train[:-1])
Xva, yva, _ = load(a.train[-1:])  # last train seed = validation (by-seed, no leakage)
Xte, yte, lte = load(a.test)
model = tf.keras.models.load_model(OUT / "models/CNN-LSTM.keras")
res = {"zero_shot": report(model, Xte, yte, lte)}
sw = (1.0 / ltr.map(ltr.value_counts())).to_numpy().copy()  # every label (benign + 4 attacks) carries equal total weight
sw *= len(sw) / sw.sum()
model.compile(optimizer=tf.keras.optimizers.Adam(a.lr), loss=model.loss, metrics=["accuracy"])
model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=a.epochs, batch_size=256,
          sample_weight=sw, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)])
res["fine_tuned"] = report(model, Xte, yte, lte)
res["setup"] = vars(a) | {"n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte)}
model.save(HERE / "results/CNN-LSTM_sim_ft.keras")
json.dump(res, open(HERE / "results/finetune.json", "w"), indent=2, default=float)
print(json.dumps(res, indent=2, default=float))
