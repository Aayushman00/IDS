"""Test-set metrics and every figure of the report (300-dpi PNGs).

Each ``plot_*`` function takes ``slide=True`` to re-render the same figure at
16:9 with large fonts into ``outputs/figures/slides``.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import matplotlib

matplotlib.use("Agg")  # headless: works in WSL/servers; notebooks display the PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, log_loss, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from config import Config
from preprocessing import label_family

log = logging.getLogger(__name__)

# ---- one palette for the whole report --------------------------------------
BENIGN_C, MALICIOUS_C = "#2A7AB0", "#D1495B"
TRAIN_C, VAL_C = "#2A7AB0", "#E08E2B"
METRIC_C = {"Accuracy": "#2A7AB0", "Precision": "#3C9D5D", "Recall": "#E08E2B", "F1": "#8E5DB0"}
MODEL_C = {"CNN-LSTM": "#D1495B", "CNN": "#2A7AB0", "LSTM": "#3C9D5D", "MLP": "#E08E2B",
           "RandomForest": "#7A7A7A", "Majority (always malicious)": "#C9C9C9"}
MAJORITY = "Majority (always malicious)"
PAPER_C = "#444444"


@contextmanager
def style(slide: bool):
    """Readable defaults; much larger fonts for slide versions."""
    base = 20 if slide else 11
    with plt.rc_context({"font.size": base, "axes.titlesize": base + 2,
                         "axes.labelsize": base, "legend.fontsize": base - 2,
                         "xtick.labelsize": base - 2, "ytick.labelsize": base - 2,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.3}):
        yield


def _fig(slide: bool, size=(8, 5), **kw):
    return plt.subplots(figsize=(16, 9) if slide else size, **kw)


def save(fig, name: str, cfg: Config, slide: bool = False) -> Path:
    """tight_layout + 300 dpi; stamps SYNTHETIC on smoke-test figures."""
    if cfg.synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC DATA - NOT REAL RESULTS", ha="center", va="center",
                 fontsize=28, color="red", alpha=0.25, rotation=25)
    fig.tight_layout()
    path = (cfg.slide_dir if slide else cfg.fig_dir) / f"{name}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _binary_counts(labels: pd.Series | dict) -> tuple[int, int]:
    s = pd.Series(labels) if isinstance(labels, dict) else labels
    fam = pd.Series(s.index.map(label_family), index=s.index)
    benign = int(s[fam == "Benign"].sum())
    return benign, int(s.sum()) - benign


# ============================================================================
# Metrics
# ============================================================================
def timed_predict(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Predict once as warm-up (graph tracing), then time a full pass."""
    predict(X[: min(len(X), 1024)])
    t0 = time.perf_counter()
    proba = predict(X)
    dt = time.perf_counter() - t0
    return proba, {"inference_ms_per_sample": 1000 * dt / len(X),
                   "throughput_samples_per_s": len(X) / dt}


def compute_metrics(y: np.ndarray, proba: np.ndarray, class_names: list[str],
                    threshold: float = 0.5) -> dict:
    """All test metrics. Binary: 'Malicious' (1) is the positive class."""
    m: dict = {}
    if len(class_names) == 2:
        score = proba[:, 1]
        pred = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        m.update(
            accuracy=accuracy_score(y, pred),
            precision=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred, zero_division=0),
            f1=f1_score(y, pred, zero_division=0),
            roc_auc=roc_auc_score(y, score) if len(np.unique(y)) == 2 else float("nan"),
            pr_auc=average_precision_score(y, score),
            tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
            fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1),
            specificity=tn / max(tn + fp, 1),
            # benign = the minority, negative class: the numbers accuracy hides
            benign_precision=precision_score(y, pred, pos_label=0, zero_division=0),
            benign_recall=recall_score(y, pred, pos_label=0, zero_division=0),
            benign_f1=f1_score(y, pred, pos_label=0, zero_division=0),
        )
    else:
        pred = proba.argmax(1)
        m.update(accuracy=accuracy_score(y, pred))
        try:
            m["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                         labels=list(range(len(class_names))))
        except ValueError:  # a class missing from the test split
            m["roc_auc"] = float("nan")
    for avg in ("macro", "weighted"):
        m[f"precision_{avg}"] = precision_score(y, pred, average=avg, zero_division=0)
        m[f"recall_{avg}"] = recall_score(y, pred, average=avg, zero_division=0)
        m[f"f1_{avg}"] = f1_score(y, pred, average=avg, zero_division=0)
    m["mcc"] = matthews_corrcoef(y, pred)
    m["log_loss"] = log_loss(y, np.clip(proba, 1e-7, 1), labels=list(range(len(class_names))))
    return {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in m.items()}


def majority_proba(n: int, n_classes: int) -> np.ndarray:
    """'Always predict malicious' baseline (class 1 in binary; most frequent otherwise)."""
    p = np.zeros((n, n_classes), np.float32)
    p[:, 1 if n_classes == 2 else 0] = 1.0
    return p


def predictions(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba[:, 1] >= threshold).astype(int) if proba.shape[1] == 2 else proba.argmax(1)


def save_classification_report(y, pred, class_names: list[str], cfg: Config, name: str) -> None:
    labels = list(range(len(class_names)))
    txt = classification_report(y, pred, labels=labels, target_names=class_names, digits=4,
                                zero_division=0)
    (cfg.metrics_dir / f"classification_report_{name}.txt").write_text(txt)
    rep = classification_report(y, pred, labels=labels, target_names=class_names,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(cfg.metrics_dir / f"classification_report_{name}.csv")


def save_results(results: dict, cfg: Config) -> None:
    """results.json (everything) + results.csv (one row per model)."""
    (cfg.metrics_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    rows = [{"model": k, **{mk: mv for mk, mv in v.items() if not isinstance(mv, (dict, list))}}
            for k, v in results["models"].items()]
    pd.DataFrame(rows).to_csv(cfg.metrics_dir / "results.csv", index=False)


# ============================================================================
# 1-5  Data exploration
# ============================================================================
def plot_class_distribution(population: pd.Series, sample_labels: pd.Series,
                            train_before: dict, train_after: dict, cfg: Config) -> Path:
    """Fig 1: benign vs malicious from full data -> sample -> balanced train."""
    panels = [("Full dataset (all CSV rows read)", _binary_counts(population)),
              ("After sampling + cleaning", _binary_counts(sample_labels.value_counts())),
              ("Train split (before balancing)", (train_before.get("Benign", 0),
                                                 sum(v for k, v in train_before.items() if k != "Benign"))),
              (f"Train split (after '{cfg.balance}')", (train_after.get("Benign", 0),
                                                       sum(v for k, v in train_after.items() if k != "Benign")))]
    with style(False):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        for ax, (title, (b, m)) in zip(axes, panels):
            bars = ax.bar(["Benign", "Malicious"], [b, m], color=[BENIGN_C, MALICIOUS_C])
            tot = max(b + m, 1)
            for bar, v in zip(bars, (b, m)):
                ax.annotate(f"{v:,.0f}\n({100 * v / tot:.1f}%)", (bar.get_x() + bar.get_width() / 2, v),
                            ha="center", va="bottom", fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Rows (weighted for class_weight)" if "after" in title and cfg.balance == "class_weight" else "Rows")
            ax.margins(y=0.25)
        fig.suptitle("Fig 1 - Class distribution: benign vs malicious")
        return save(fig, "01_class_distribution", cfg)


def plot_top_attacks(population: pd.Series, cfg: Config) -> Path:
    """Fig 2: 15 most frequent attack sub-types (log scale: DDoS dwarfs web attacks)."""
    attacks = population[[label_family(i) != "Benign" for i in population.index]].nlargest(15)[::-1]
    with style(False):
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = [plt.cm.tab10(["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
                               .index(label_family(i))) for i in attacks.index]
        ax.barh(attacks.index, attacks.values, color=colors)
        ax.set_xscale("log")
        ax.set_xlabel("Rows (log scale)")
        ax.set_title("Fig 2 - Top-15 attack types in CICIoT2023")
        return save(fig, "02_top15_attack_types", cfg)


def _top_corr_features(df: pd.DataFrame, features: list[str], k: int) -> list[str]:
    y = (df["label"].map(label_family) != "Benign").astype(float)
    corr = df[features].astype(float).corrwith(y).abs().fillna(0)
    return corr.nlargest(k).index.tolist()


def plot_correlation(df_train: pd.DataFrame, features: list[str], cfg: Config) -> Path:
    """Fig 3: Pearson correlation of the 20 features most correlated with the label."""
    top = _top_corr_features(df_train, features, 20)
    corr = df_train[top].astype(float).corr()
    with style(False):
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, ax=ax,
                    cbar_kws={"label": "Pearson r"}, linewidths=0.3)
        ax.grid(False)
        ax.set_title("Fig 3 - Correlation of top-20 label-correlated features (train split)")
        return save(fig, "03_feature_correlation", cfg)


def plot_feature_boxplots(df_train: pd.DataFrame, features: list[str], cfg: Config) -> Path:
    """Fig 4: log1p-scaled boxplots, benign vs malicious, for 6 key features."""
    top = _top_corr_features(df_train, features, 6)
    sub = df_train.sample(min(len(df_train), 50_000), random_state=cfg.seed)
    mal = sub["label"].map(label_family) != "Benign"
    with style(False):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, f in zip(axes.ravel(), top):
            vals = np.log1p(np.clip(sub[f].astype(float), 0, None))
            bp = ax.boxplot([vals[~mal], vals[mal]], tick_labels=["Benign", "Malicious"],
                            patch_artist=True, showfliers=False, widths=0.6)
            for patch, c in zip(bp["boxes"], (BENIGN_C, MALICIOUS_C)):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            ax.set_title(f)
            ax.set_ylabel("log(1 + value)")
        fig.suptitle("Fig 4 - Key feature distributions, benign vs malicious (train split)")
        return save(fig, "04_feature_boxplots", cfg)


def plot_missing_inf(report: dict, cfg: Config) -> Path:
    """Fig 5: NaN and inf counts per column, measured before cleaning."""
    nan = pd.Series(report["nan_per_column"])
    inf = pd.Series(report["inf_per_column"])
    nan_only = (nan - inf).clip(lower=0)  # inf were converted to NaN before counting NaN
    with style(False):
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(nan))
        ax.bar(x, nan_only.values, color=BENIGN_C, label="NaN")
        ax.bar(x, inf.values, bottom=nan_only.values, color=MALICIOUS_C, label="+/-inf")
        ax.set_xticks(x, nan.index, rotation=90, fontsize=8)
        ax.set_ylabel("Rows affected")
        if (nan.sum() + inf.sum()) == 0:
            ax.text(0.5, 0.5, "No NaN or inf values found in the sample", transform=ax.transAxes,
                    ha="center", fontsize=14)
            ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Fig 5 - Missing / infinite values before cleaning "
                     f"(duplicates removed: {report.get('duplicates_removed', 0):,})")
        return save(fig, "05_missing_inf_summary", cfg)


# ============================================================================
# 6-8  Training behaviour
# ============================================================================
def plot_history(hist: pd.DataFrame, metric: str, cfg: Config, slide: bool = False) -> Path:
    """Fig 6 (accuracy) / Fig 7 (loss)."""
    num = {"accuracy": 6, "loss": 7}[metric]
    with style(slide):
        fig, ax = _fig(slide)
        ep = np.arange(1, len(hist) + 1)
        ax.plot(ep, hist[metric], "-o", color=TRAIN_C, label=f"Train {metric}", ms=4)
        ax.plot(ep, hist[f"val_{metric}"], "-s", color=VAL_C, label=f"Validation {metric}", ms=4)
        best = int(hist["val_loss"].idxmin()) + 1
        ax.axvline(best, ls="--", color="grey", lw=1, label=f"Best epoch ({best})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Fig {num} - CNN-LSTM training vs validation {metric}")
        ax.legend()
        return save(fig, f"{num:02d}_training_{metric}", cfg, slide)


def plot_lr(hist: pd.DataFrame, cfg: Config) -> Path:
    """Fig 8: learning rate per epoch (steps show ReduceLROnPlateau firing)."""
    with style(False):
        fig, ax = _fig(False)
        ax.step(np.arange(1, len(hist) + 1), hist["lr"], where="post", color=METRIC_C["F1"])
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate (log)")
        n_drops = int((np.diff(hist["lr"]) < 0).sum())
        ax.set_title(f"Fig 8 - Learning-rate schedule (ReduceLROnPlateau fired {n_drops}x)")
        return save(fig, "08_learning_rate", cfg)


# ============================================================================
# 9-16  Test performance
# ============================================================================
def plot_confusion(y, pred, class_names: list[str], cfg: Config, normalize: bool,
                   slide: bool = False, name: Optional[str] = None) -> Path:
    """Fig 9 (counts) / Fig 10 (row-normalised %). Binary cells carry TP/TN/FP/FN."""
    cm = confusion_matrix(y, pred, labels=range(len(class_names)))
    big = len(class_names) > 2
    if normalize:
        data = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fmt = ".1f" if not big else ".0f"
    else:
        data, fmt = cm, "d"
    annot = True
    if not big:
        tags = np.array([["TN", "FP"], ["FN", "TP"]])
        annot = np.array([[f"{tags[i, j]}\n{data[i, j]:{',' if not normalize else '.2f'}}"
                           + ("%" if normalize else "") for j in range(2)] for i in range(2)])
        fmt = ""
    with style(slide):
        size = (7, 6) if not big else (max(10, 0.45 * len(class_names)),) * 2
        fig, ax = _fig(slide, size)
        sns.heatmap(data, annot=annot, fmt=fmt, cmap="Blues", ax=ax, cbar=True,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"fontsize": 6 if len(class_names) > 12 else (22 if slide else 12)},
                    square=not slide)
        ax.grid(False)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        num = 10 if normalize else 9
        kind = "normalised (% of true class)" if normalize else "raw counts"
        ax.set_title(f"Fig {num} - CNN-LSTM confusion matrix, {kind} (test set)")
        return save(fig, name or f"{num:02d}_confusion_matrix_{'normalized' if normalize else 'counts'}",
                    cfg, slide)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], cfg: Config, slide: bool = False) -> Path:
    """Fig 11: ROC for every binary model (main model drawn thick)."""
    with style(slide):
        fig, ax = _fig(slide, (7, 6))
        for name, (y, s) in curves.items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=3 if name == "CNN-LSTM" else 1.5, color=MODEL_C.get(name),
                    label=f"{name} (AUC = {roc_auc_score(y, s):.4f})")
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Fig 11 - ROC curve (test set, positive = malicious)")
        ax.legend(loc="lower right")
        return save(fig, "11_roc_curve", cfg, slide)


def plot_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], cfg: Config) -> Path:
    """Fig 12: precision-recall; baseline = malicious prevalence."""
    with style(False):
        fig, ax = _fig(False, (7, 6))
        for name, (y, s) in curves.items():
            p, r, _ = precision_recall_curve(y, s)
            ax.plot(r, p, lw=3 if name == "CNN-LSTM" else 1.5, color=MODEL_C.get(name),
                    label=f"{name} (AP = {average_precision_score(y, s):.4f})")
        y0 = next(iter(curves.values()))[0]
        ax.axhline(y0.mean(), ls="--", color="grey", lw=1, label=f"Prevalence ({y0.mean():.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Fig 12 - Precision-Recall curve (test set)")
        ax.legend(loc="lower left")
        return save(fig, "12_pr_curve", cfg)


def plot_metric_bars(m: dict, cfg: Config, slide: bool = False) -> Path:
    """Fig 13: Accuracy / Precision / Recall / F1 of the CNN-LSTM."""
    binary = "precision" in m
    names = ["Accuracy", "Precision", "Recall", "F1"]
    keys = (["accuracy", "precision", "recall", "f1"] if binary else
            ["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    wkeys = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    with style(slide):
        fig, ax = _fig(slide)
        x = np.arange(4)
        w = 0.38
        b1 = ax.bar(x - w / 2, [100 * m[k] for k in keys], w, color=[METRIC_C[n] for n in names],
                    label="Attack = positive" if binary else "Macro avg")
        b2 = ax.bar(x + w / 2, [100 * m[k] for k in wkeys], w, color=[METRIC_C[n] for n in names],
                    alpha=0.45, hatch="//", label="Support-weighted avg over both classes")
        ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=18 if slide else 9)
        ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=18 if slide else 9)
        ax.set_xticks(x, names)
        lo = min(100 * m[k] for k in keys + wkeys)
        ax.set_ylim(max(0, lo - 5), 100.8)
        ax.set_ylabel("Score (%)")
        ax.set_title("Fig 13 - CNN-LSTM test-set metrics")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
        return save(fig, "13_metric_bars", cfg, slide)


def plot_proba_hist(y, score, threshold: float, cfg: Config) -> Path:
    """Fig 14: P(malicious) per true class, log y so the benign tail is visible."""
    with style(False):
        fig, ax = _fig(False)
        bins = np.linspace(0, 1, 51)
        ax.hist(score[y == 0], bins, color=BENIGN_C, alpha=0.7, label="True benign")
        ax.hist(score[y == 1], bins, color=MALICIOUS_C, alpha=0.6, label="True malicious")
        ax.axvline(threshold, color="black", ls="--", label=f"Threshold = {threshold}")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted P(malicious)")
        ax.set_ylabel("Test samples (log)")
        ax.set_title("Fig 14 - Prediction probability distribution (test set)")
        ax.legend()
        return save(fig, "14_probability_histogram", cfg)


def plot_threshold_sweep(y, score, threshold: float, cfg: Config) -> Path:
    """Fig 15: precision / recall / F1 / benign recall as the threshold moves."""
    ts = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in ts:
        p = (score >= t).astype(int)
        rows.append((precision_score(y, p, zero_division=0), recall_score(y, p, zero_division=0),
                     f1_score(y, p, zero_division=0), recall_score(1 - y, 1 - p, zero_division=0)))
    P, R, F, S = np.array(rows).T
    best = ts[F.argmax()]
    with style(False):
        fig, ax = _fig(False)
        ax.plot(ts, P, color=METRIC_C["Precision"], label="Precision (malicious)")
        ax.plot(ts, R, color=METRIC_C["Recall"], label="Recall (malicious)")
        ax.plot(ts, F, color=METRIC_C["F1"], lw=2.5, label="F1 (malicious)")
        ax.plot(ts, S, color=BENIGN_C, ls=":", label="Specificity (benign recall)")
        ax.axvline(threshold, color="black", ls="--", lw=1, label=f"Used threshold {threshold}")
        ax.axvline(best, color=METRIC_C["F1"], ls="--", lw=1, label=f"Best-F1 threshold {best:.2f}")
        ax.set_xlabel("Decision threshold on P(malicious)")
        ax.set_ylabel("Score")
        ax.set_title("Fig 15 - Threshold sweep (test set)")
        ax.legend(loc="lower left", fontsize=9)
        return save(fig, "15_threshold_sweep", cfg)


def plot_per_class_f1(y, pred, class_names: list[str], cfg: Config) -> Path:
    """Fig 16a (multiclass only): F1 per class."""
    f1 = f1_score(y, pred, average=None, labels=range(len(class_names)), zero_division=0)
    order = np.argsort(f1)
    with style(False):
        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(class_names))))
        ax.barh(np.array(class_names)[order], f1[order], color=METRIC_C["F1"])
        ax.set_xlabel("F1-score")
        ax.set_xlim(0, 1)
        ax.set_title("Fig 16 - Per-class F1 (test set)")
        return save(fig, "16_per_class_f1", cfg)


# ============================================================================
# 17-22  Comparison & analysis
# ============================================================================
def plot_model_comparison(table: pd.DataFrame, cfg: Config, slide: bool = False) -> Path:
    """Fig 17: grouped bars, one group per metric, one bar per model."""
    metrics = ["accuracy", "precision", "recall", "f1"] if "precision" in table else \
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    with style(slide):
        fig, ax = _fig(slide, (11, 5.5))
        n = len(table)
        w = 0.8 / n
        x = np.arange(len(metrics))
        for i, (name, row) in enumerate(table.iterrows()):
            bars = ax.bar(x + (i - (n - 1) / 2) * w, [100 * row[m] for m in metrics], w,
                          color=MODEL_C.get(name), label=name)
            ax.bar_label(bars, fmt="%.2f", fontsize=14 if slide else 7, rotation=90, padding=2)
        lo = min(100 * table[metrics].min().min(), 99)
        ax.set_ylim(max(0, lo - 3), 104.5)  # headroom for the rotated value labels
        ax.set_yticks([t for t in ax.get_yticks() if max(0, lo - 3) <= t <= 100])  # no ticks above 100 %
        ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1"])
        ax.set_ylabel("Score (%)")
        ax.set_title("Fig 17 - Model comparison on the test set")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
        return save(fig, "17_model_comparison", cfg, slide)


def plot_cost_comparison(table: pd.DataFrame, cfg: Config) -> Path:
    """Fig 19: latency per sample and on-disk model size per model."""
    table = table[table["model_size_mb"].notna()]  # majority baseline has no model
    with style(False):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
        colors = [MODEL_C.get(n) for n in table.index]
        b = a1.bar(table.index, 1000 * table["inference_ms_per_sample"], color=colors)
        a1.bar_label(b, fmt="%.2f")
        a1.set_ylabel("Inference time (µs / sample, batched)")
        a1.set_title("Inference latency")
        b = a2.bar(table.index, table["model_size_mb"], color=colors)
        a2.bar_label(b, fmt="%.2f")
        a2.set_ylabel("Model size on disk (MB)")
        a2.set_title("Model size")
        for a in (a1, a2):
            a.tick_params(axis="x", rotation=20)
        fig.suptitle("Fig 19 - Inference cost comparison (test set, GPU batch 4096; RF on CPU)")
        return save(fig, "19_inference_cost_comparison", cfg)


def error_by_subtype(sub_labels: np.ndarray, y, pred) -> pd.DataFrame:
    """Per original sub-label: share of test rows the model got wrong.

    For attack sub-types in the binary task this is exactly the FN rate
    (attack predicted benign); for benign it is the FP rate.
    """
    df = pd.DataFrame({"sub": sub_labels, "wrong": np.asarray(y) != np.asarray(pred)})
    out = df.groupby("sub")["wrong"].agg(["mean", "sum", "count"])
    out.columns = ["error_rate", "errors", "n"]
    return out.sort_values("error_rate", ascending=False)


def plot_error_analysis(err: pd.DataFrame, cfg: Config) -> Path:
    """Fig 20: FN rate per attack sub-type (benign bar = FP rate)."""
    top = err.head(20)[::-1]
    colors = [BENIGN_C if label_family(i) == "Benign" else MALICIOUS_C for i in top.index]
    with style(False):
        fig, ax = plt.subplots(figsize=(10, 7))
        bars = ax.barh(top.index, 100 * top["error_rate"], color=colors)
        ax.bar_label(bars, labels=[f"{r:.1f}% ({e:,}/{n:,})" for r, e, n in
                                   zip(100 * top.error_rate, top.errors, top.n)], fontsize=8, padding=2)
        ax.set_xlabel("Misclassified test rows (%)  —  attacks: FN rate, benign: FP rate")
        ax.set_title("Fig 20 - Error analysis per original traffic sub-type (top 20)")
        ax.margins(x=0.25)
        return save(fig, "20_error_by_attack_type", cfg)


def plot_tsne(emb: np.ndarray, y: np.ndarray, class_names: list[str], cfg: Config) -> Path:
    """Fig 21: t-SNE of the last LSTM layer's output (what the head sees)."""
    z = TSNE(n_components=2, init="pca", random_state=cfg.seed, perplexity=30).fit_transform(emb)
    with style(False):
        fig, ax = plt.subplots(figsize=(8, 7))
        if len(class_names) == 2:
            for k, c in ((1, MALICIOUS_C), (0, BENIGN_C)):  # benign on top: minority class
                ax.scatter(*z[y == k].T, s=4, c=c, alpha=0.6, label=class_names[k])
        else:
            for k, n in enumerate(class_names):
                if (y == k).any():
                    ax.scatter(*z[y == k].T, s=4, alpha=0.6, label=n)
        ax.legend(markerscale=4, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Fig 21 - t-SNE of LSTM embeddings ({len(y):,} test samples)")
        return save(fig, "21_tsne_lstm_embeddings", cfg)


def plot_cv(cv: pd.DataFrame, cfg: Config) -> Path:
    """Fig 22: fold-to-fold spread of the CNN-LSTM."""
    cols = ["accuracy", "precision", "recall", "f1"]
    with style(False):
        fig, ax = _fig(False)
        bp = ax.boxplot([100 * cv[c] for c in cols], tick_labels=[c.capitalize() for c in cols],
                        patch_artist=True, widths=0.5)
        for patch, n in zip(bp["boxes"], METRIC_C):
            patch.set_facecolor(METRIC_C[n])
            patch.set_alpha(0.6)
        for i, c in enumerate(cols, 1):
            ax.scatter(np.full(len(cv), i), 100 * cv[c], color="black", s=12, zorder=3)
            ax.text(i, 100 * cv[c].min(), f"{100 * cv[c].mean():.2f}±{100 * cv[c].std():.2f}",
                    ha="center", va="top", fontsize=9)
        ax.set_ylabel("Score (%)")
        ax.set_title(f"Fig 22 - {len(cv)}-fold stratified CV of CNN-LSTM (mean ± std)")
        return save(fig, "22_cross_validation", cfg)
