"""Run the trained CNN-LSTM on simulated windows and report detection metrics.
usage: python ns3_sim/ids_infer.py features.csv metrics.json [outputs_dir]   (from iot_ids_cnn_lstm/; default outputs/)
Same preprocessing as training: feature order from meta.json, saved MinMax scaler, input (n, F, 1)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def main(feat_csv, out_json, outputs="outputs", threshold=0.5):
    OUT = ROOT / outputs  # model, scaler and feature list must come from the same run
    import tensorflow as tf  # late: keeps import errors/--help fast
    df = pd.read_csv(feat_csv)
    cols = json.load(open(OUT / "cache/meta.json"))["feature_names"]
    X = joblib.load(OUT / "models/scaler.joblib").transform(df[cols].to_numpy(np.float32))
    oor = float(((X < -0.01) | (X > 1.01)).any(axis=1).mean())  # share of windows outside the training range
    model = tf.keras.models.load_model(OUT / "models/CNN-LSTM.keras")
    p = model.predict(X[..., None], batch_size=1024, verbose=0)[:, 1]
    y, yh = (df.label != "Benign").to_numpy().astype(int), (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    res = dict(n=len(df), accuracy=(tp + tn) / len(df), precision=precision_score(y, yh, zero_division=0),
               recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
               fpr=fp / max(fp + tn, 1), roc_auc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else None,
               confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)), out_of_range_share=oor,
               detected_share_by_label={k: float(g.mean()) for k, g in pd.Series(yh == 1).groupby(df.label)})
    df["p_malicious"] = p
    df.to_csv(Path(feat_csv).with_name(f"predictions_{outputs}.csv" if outputs != "outputs" else "predictions.csv"), index=False)
    json.dump(res, open(out_json, "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:4])
