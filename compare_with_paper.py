"""Our CNN-LSTM vs the numbers reported by Gueriani et al. (PAIS 2024).

Only paper values present in config.PAPER_RESULTS (not None) are compared;
missing ones are shown as "n/a" and never filled in by this script.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import PAPER_RESULTS, PAPER_SETUP, Config
from evaluate import MAJORITY, MODEL_C, PAPER_C, _fig, save, style

log = logging.getLogger(__name__)

# (paper key, our key, label, shown in percent?)
ROWS = [
    ("accuracy", "accuracy", "Accuracy (%)", True),
    ("precision", "precision_weighted", "Precision, support-weighted (%)", True),
    ("recall", "recall_weighted", "Recall, support-weighted (%)", True),
    ("f1", "f1_weighted", "F1, support-weighted (%)", True),
    (None, "precision", "Precision, malicious = positive (%)", True),
    (None, "recall", "Recall / detection rate, malicious (%)", True),
    (None, "f1", "F1, malicious (%)", True),
    ("benign_precision", "benign_precision", "Benign precision (%)", True),
    ("benign_recall", "benign_recall", "Benign recall = specificity (%)", True),
    ("benign_f1", "benign_f1", "Benign F1 (%)", True),
    ("fpr", "fpr", "False positive rate (%)", True),
    (None, "fnr", "False negative rate (%)", True),
    (None, "mcc", "Matthews corr. coef.", False),
    ("roc_auc", "roc_auc", "ROC-AUC", False),
    (None, "pr_auc", "PR-AUC (average precision)", False),
    ("loss", "log_loss", "Cross-entropy loss", False),
]


def _fmt(v: Optional[float], pct: bool) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(100 * v if pct else v, 4)


def comparison_table(ours: dict, majority: Optional[dict] = None) -> pd.DataFrame:
    """Paper vs ours vs the always-malicious baseline, with deviation in pp."""
    recs = []
    for pk, ok, label, pct in ROWS:
        if ok not in ours:
            continue
        paper = PAPER_RESULTS.get(pk) if pk else None
        mine = _fmt(ours[ok], pct)
        rec = {"metric": label, "paper": paper, "ours_cnn_lstm": mine,
               "majority_baseline": _fmt(majority.get(ok), pct) if majority else None,
               "abs_diff_vs_paper": None, "rel_diff_vs_paper_%": None}
        if paper is not None and mine is not None:
            rec["abs_diff_vs_paper"] = round(mine - paper, 4)
            rec["rel_diff_vs_paper_%"] = round(100 * (mine - paper) / paper, 3)
        recs.append(rec)
    return pd.DataFrame(recs)


def plot_paper_comparison(ours: dict, cfg: Config, slide: bool = False,
                          majority: Optional[dict] = None) -> Path:
    """Fig 18: our A/P/R/F1 next to the paper's (only where the paper value is known),
    with the paper's 98.42 % accuracy as a dashed reference line."""
    keys = [("accuracy", "accuracy", "Accuracy"), ("precision", "precision_weighted", "Precision\n(weighted)"),
            ("recall", "recall_weighted", "Recall\n(weighted)"), ("f1", "f1_weighted", "F1\n(weighted)"),
            (None, "benign_recall", "Benign recall"), (None, "benign_f1", "Benign F1")]
    with style(slide):
        fig, ax = _fig(slide, (11, 5.8))
        x = np.arange(len(keys))
        w = 0.27
        mine = [100 * ours[o] for _, o, _ in keys]
        b2 = ax.bar(x, mine, w, color=MODEL_C["CNN-LSTM"], label="Our CNN-LSTM reproduction")
        ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=16 if slide else 9)
        paper_vals = [PAPER_RESULTS.get(p) if p else None for p, _, _ in keys]
        px = [i for i, v in enumerate(paper_vals) if v is not None]
        if px:
            b1 = ax.bar(x[px] - w, [paper_vals[i] for i in px], w, color=PAPER_C,
                        label="Paper (Gueriani et al., 2024)")
            ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=16 if slide else 9)
        for i, v in enumerate(paper_vals):
            if v is None:
                ax.text(x[i] - w, 0.015, "paper:\nn/a", ha="center", va="bottom",
                        fontsize=13 if slide else 8, color=PAPER_C, transform=ax.get_xaxis_transform())
        if majority:
            b3 = ax.bar(x + w, [100 * majority[o] for _, o, _ in keys], w,
                        color=MODEL_C[MAJORITY], label=MAJORITY)
            ax.bar_label(b3, fmt="%.1f", padding=2, fontsize=16 if slide else 8)
        ax.axhline(PAPER_RESULTS["accuracy"], ls="--", color="black", lw=1.2,
                   label=f"Paper accuracy {PAPER_RESULTS['accuracy']}%")
        ax.set_ylim(0, 112)
        ax.set_xticks(x, [k[2] for k in keys])
        ax.set_ylabel("Score (%)")
        ax.set_title("Fig 18 - Our CNN-LSTM vs the paper (CICIoT2023, binary, test set)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        return save(fig, "18_paper_comparison", cfg, slide)


def deviation_analysis(ours: dict, run_info: dict, table: pd.DataFrame,
                       majority: Optional[dict] = None) -> str:
    """Plain-text list of every known difference to the paper's setup."""
    acc_gap = 100 * ours["accuracy"] - PAPER_RESULTS["accuracy"]
    lines = [
        "DEVIATION ANALYSIS - CNN-LSTM reproduction of Gueriani, Kheddar & Mazari (IEEE PAIS 2024)",
        "=" * 88,
        f"Accuracy: ours {100 * ours['accuracy']:.2f}% vs paper {PAPER_RESULTS['accuracy']}% "
        f"-> {acc_gap:+.2f} percentage points ({100 * acc_gap / PAPER_RESULTS['accuracy']:+.2f}% relative).",
        "Only the paper's accuracy is used as a reference; its other metrics are left as None in "
        "config.PAPER_RESULTS until checked against the published paper.",
    ]
    if majority:
        lines.append(
            f"Context: always predicting 'malicious' already scores {100 * majority['accuracy']:.2f}% "
            f"accuracy on our test split (benign recall 0%, MCC 0). The model's real value shows in "
            f"benign recall {100 * ours['benign_recall']:.2f}%, FPR {100 * ours['fpr']:.2f}%, "
            f"MCC {ours['mcc']:.4f}.")
    lines += [
        "",
        "Comparison table:",
        table.to_string(index=False),
        "",
        "Plausible causes of any gap (differences between the two setups):",
        f"1. Sample size vs full dataset: we trained on a {run_info['rows_sampled']:,}-row stratified "
        f"random sample of the {run_info.get('population_rows', 0):,} rows available "
        f"({run_info['rows_after_cleaning']:,} after cleaning). The paper used about "
        f"{PAPER_SETUP['train_val_rows']:,} train/val rows plus a separate "
        f"{PAPER_SETUP['test_rows']:,}-row test subset taken from other CSV files.",
        f"2. Split ratio: ours {run_info['split']}; paper 80/20 train/validation plus a separate "
        "test subset. Different test sets mean the numbers are not strictly paired.",
        f"3. Duplicates: we dropped {run_info['duplicates_removed']:,} exact duplicate rows BEFORE "
        "splitting so no identical flow appears in both train and test; the paper does not mention "
        "de-duplication. Duplicates shared across splits usually inflate test accuracy.",
        f"4. Hyperparameters assumed, not taken from the paper: batch size {run_info['batch_size']}, "
        f"learning rate {run_info['learning_rate']}, pooling size 2, ReLU activations, "
        "EarlyStopping(patience 5) + ReduceLROnPlateau. Paper: Adam, 25 epochs. "
        f"We ran {run_info.get('epochs_run', '?')} epochs; best val_loss at epoch "
        f"{run_info.get('best_epoch', '?')}"
        + (" (early stopping triggered)." if run_info.get("early_stopped") else "."),
        f"5. Class imbalance: '{run_info['balance']}' (weights {run_info.get('class_weight')}). "
        "The paper does not mention re-balancing. Weighting the ~2-3% benign class trades some "
        "accuracy for much higher benign recall / lower FPR.",
        f"6. Features: the paper states 45 inputs; our release has {run_info['n_features_raw']}, "
        f"{run_info['n_features_used']} used after dropping train-constant columns "
        f"{run_info['dropped_constant'] or '(none)'}.",
        f"7. Scaling: {run_info['scaler']} fitted on train only (paper: not stated).",
        f"8. Task: {run_info['task']} (paper: binary benign vs malicious) - same task.",
        f"9. Seed {run_info['seed']}; cuDNN LSTM/conv kernels are not bit-exact between GPUs, so "
        "re-runs can differ slightly.",
        f"10. Dataset release: {run_info['dataset_release']}. Obtained from a public mirror of the "
        "official CIC files (train/validation/test CSVs pooled and re-split by us).",
        "11. Metric convention: 'Precision/Recall/F1' in the paper may be weighted or "
        "attack-positive; we report both so the right one can be compared once the paper values "
        "are filled in.",
    ]
    return "\n".join(lines)


def run(ours: dict, run_info: dict, cfg: Config, majority: Optional[dict] = None) -> pd.DataFrame:
    table = comparison_table(ours, majority)
    table.to_csv(cfg.metrics_dir / "paper_comparison.csv", index=False)
    (cfg.metrics_dir / "paper_comparison.md").write_text(
        table.to_markdown(index=False, missingval="n/a"), encoding="utf-8")
    if cfg.task == "binary":
        plot_paper_comparison(ours, cfg, majority=majority)
        plot_paper_comparison(ours, cfg, slide=True, majority=majority)
    (cfg.metrics_dir / "deviation_analysis.txt").write_text(
        deviation_analysis(ours, run_info, table, majority), encoding="utf-8")
    log.info("Paper comparison:\n%s", table.to_string(index=False))
    return table


if __name__ == "__main__":
    # Re-generate the comparison from a finished run without retraining.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    c = Config()
    res = json.loads((c.metrics_dir / "results.json").read_text())
    run(res["models"]["CNN-LSTM"], res["run_info"], c, res["models"].get(MAJORITY))
