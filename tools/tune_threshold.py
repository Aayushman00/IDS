"""Choose the decision threshold on the VALIDATION split, then report TEST metrics.

The threshold is picked by maximising MCC on validation predictions only
(the test split is never used for the choice, so the test numbers stay an
honest held-out estimate). Writes threshold_tuning.json/.md and
figures/24_threshold_tuning.png; results.json is not modified.

usage: python tools/tune_threshold.py [--output_dir outputs] [--sample 1400000] [--balance class_weight]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import matthews_corrcoef  # noqa: E402

import evaluate as ev  # noqa: E402
from config import Config  # noqa: E402
from preprocessing import load_cache  # noqa: E402
from train import predict_proba, setup_gpu  # noqa: E402

KEYS = ["accuracy", "precision", "recall", "f1", "benign_precision", "benign_recall", "benign_f1",
        "fpr", "fnr", "mcc", "tp", "tn", "fp", "fn"]


def sweep(y, p, thresholds):
    rows = []
    for t in thresholds:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        rows.append((t, matthews_corrcoef(y, pred), fp / max(fp + tn, 1), fn / max(fn + tp, 1)))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=Path("outputs"))
    ap.add_argument("--sample", type=int, default=1_400_000)
    ap.add_argument("--balance", default="class_weight")
    ap.add_argument("--model", default="CNN-LSTM")
    args = ap.parse_args()
    cfg = Config(output_dir=args.output_dir, sample_rows=args.sample, balance=args.balance)
    setup_gpu(cfg)
    data, _ = load_cache(cfg)
    model = keras.models.load_model(cfg.model_dir / f"{args.model}.keras")
    yv, yt = np.asarray(data.y_val), np.asarray(data.y_test)
    pv = predict_proba(model, data.X_val)[:, 1]
    pt = predict_proba(model, data.X_test)[:, 1]

    # candidates: dense grid + quantiles of the validation scores (probabilities pile up near 0/1)
    cand = np.unique(np.r_[np.linspace(0.001, 0.999, 999), np.quantile(pv, np.linspace(0, 1, 2001))])
    cand = cand[(cand > 0) & (cand < 1)]
    s = sweep(yv, pv, cand)
    best = s[np.lexsort((np.abs(s[:, 0] - 0.5), -s[:, 1]))[0]]  # max MCC, ties -> closest to 0.5
    t_best = float(best[0])

    proba_t = np.c_[1 - pt, pt]
    at_05 = ev.compute_metrics(yt, proba_t, data.class_names, 0.5)
    at_tb = ev.compute_metrics(yt, proba_t, data.class_names, t_best)
    out = {"model": args.model, "selection": "max MCC on the validation split (test never used)",
           "threshold_default": 0.5, "threshold_tuned": t_best, "val_mcc_at_tuned": float(best[1]),
           "val_mcc_at_0.5": float(sweep(yv, pv, [0.5])[0, 1]),
           "test_at_0.5": {k: at_05[k] for k in KEYS}, "test_at_tuned": {k: at_tb[k] for k in KEYS}}
    (cfg.metrics_dir / "threshold_tuning.json").write_text(json.dumps(out, indent=2))

    def row(name, m):
        return (f"| {name} | {100 * m['accuracy']:.2f} | {100 * m['recall']:.2f} | {100 * m['benign_recall']:.2f} | "
                f"{100 * m['benign_precision']:.2f} | {100 * m['fpr']:.3f} | {100 * m['fnr']:.3f} | {m['mcc']:.4f} | "
                f"{m['fp']:,} | {m['fn']:,} |")
    md = ["| Threshold (test set) | Accuracy | Attack recall | Benign recall | Benign precision | FPR | FNR | MCC | FP | FN |",
          "|---|---|---|---|---|---|---|---|---|---|",
          row("0.5 (default)", at_05), row(f"{t_best:.4f} (tuned on validation)", at_tb)]
    (cfg.metrics_dir / "threshold_tuning.md").write_text("\n".join(md) + "\n")

    fine = sweep(yv, pv, np.unique(np.r_[np.logspace(-4, 0, 300)[:-1], np.linspace(0.01, 0.99, 99)]))
    with ev.style(False):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(fine[:, 0], fine[:, 1], color=ev.METRIC_C["F1"], lw=2.5, label="Validation MCC")
        ax.plot(fine[:, 0], fine[:, 2], color=ev.BENIGN_C, ls=":", lw=2, label="Validation FPR")
        ax.plot(fine[:, 0], fine[:, 3], color=ev.MALICIOUS_C, ls="--", lw=2, label="Validation FNR")
        ax.axvline(0.5, color="black", ls="--", lw=1, label="default 0.5")
        ax.axvline(t_best, color=ev.METRIC_C["Precision"], lw=1.5, label=f"tuned {t_best:.4f}")
        ax.set_xscale("log")
        ax.set_xlabel("Decision threshold on P(malicious) (log scale)")
        ax.set_ylabel("Score / rate")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"Fig 24 - Threshold chosen on validation ({args.model}); test MCC "
                     f"{at_05['mcc']:.3f} -> {at_tb['mcc']:.3f}")
        ax.legend(loc="center left", fontsize=9)
        ev.save(fig, "24_threshold_tuning", cfg)
    print(json.dumps({"threshold": t_best, "test_mcc": [at_05["mcc"], at_tb["mcc"]],
                      "test_acc": [at_05["accuracy"], at_tb["accuracy"]],
                      "fp": [at_05["fp"], at_tb["fp"]], "fn": [at_05["fn"], at_tb["fn"]]}))


if __name__ == "__main__":
    main()
