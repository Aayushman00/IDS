"""End-to-end CNN-LSTM IDS reproduction on CICIoT2023, as resumable stages.

Stages (each saves its results to disk as soon as it finishes and is skipped
next time unless --force is given):
  prepare    CSVs -> cleaned, split, scaled float32 arrays in outputs/cache/ (+ figs 1-5)
  train      CNN-LSTM (resumable per epoch) -> model, metrics, figs 6-16, 20, 21
  baselines  CNN, LSTM, MLP, Random Forest -> metrics, figs 11, 12, 17, 19
  cv         k-fold CV on a cached subset -> fig 22
  compare    our results vs the paper -> fig 18, deviation analysis
  edge       TFLite export + latency benchmark -> fig 23

Examples:
  python main.py --smoke_test --stage all --run_baselines --cv --edge
  python main.py --stage prepare --sample 1400000
  python main.py --stage train --sample 1400000 --epochs 25      # resumes if interrupted
  python main.py --stage baselines --sample 1400000
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import joblib
import keras
import numpy as np
import pandas as pd

import compare_with_paper
import evaluate as ev
from config import Config
from data_loader import load_dataset, make_synthetic
from model import BASELINE_BUILDERS, EMBEDDING_LAYER, build_main_model
from preprocessing import (build_label_mapping, cache_is_valid, check_no_leakage, clean,
                           encode_target, load_cache, prepare, save_cache)
from runtime import log_ram, stage
from train import (cross_validate, fit_keras, fit_random_forest, predict_proba, save_summary,
                   set_seeds, setup_gpu, stratified_subset)

log = logging.getLogger("main")
STAGES = ["prepare", "train", "baselines", "cv", "compare", "edge"]


# ============================================================================
# CLI / config
# ============================================================================
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all",
                   help="all | comma list of: " + ", ".join(STAGES))
    p.add_argument("--force", action="store_true", help="re-run stages whose outputs exist")
    p.add_argument("--use_cache", action="store_true",
                   help="never read CSVs; fail if the prepared cache is missing")
    p.add_argument("--no_resume", action="store_false", dest="resume", default=None)
    p.add_argument("--data_dir", type=Path)
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--task", choices=["binary", "multiclass_8", "multiclass_34"])
    p.add_argument("--sample", type=int, help="rows to sample; 0 = all rows")
    p.add_argument("--max_files", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float, dest="learning_rate")
    p.add_argument("--architecture", choices=["paper", "generic"])
    p.add_argument("--balance", choices=["none", "class_weight", "undersample"])
    p.add_argument("--scaler", choices=["minmax", "standard"])
    p.add_argument("--threshold", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--rf_max_rows", type=int)
    p.add_argument("--cv_max_rows", type=int)
    p.add_argument("--run_baselines", action="store_true", default=None)
    p.add_argument("--cv", action="store_true", dest="run_cv", default=None)
    p.add_argument("--edge", action="store_true", dest="run_edge", default=None)
    p.add_argument("--no_tsne", action="store_false", dest="run_tsne", default=None)
    p.add_argument("--mixed_precision", action="store_true", default=None)
    p.add_argument("--deterministic", action="store_true", dest="deterministic_ops", default=None)
    p.add_argument("--smoke_test", action="store_true")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> tuple[Config, list[str], bool]:
    cfg = Config()
    if args.smoke_test:
        cfg.apply_smoke_test()
    skip = {"smoke_test", "stage", "use_cache"}
    for key, val in vars(args).items():
        if key in skip or val is None:
            continue
        if key == "sample":
            cfg.sample_rows = val or None
        else:
            setattr(cfg, key, val)
    cfg.refresh_paths()
    if args.stage == "all":
        stages = ["prepare", "train"] + (["baselines"] if cfg.run_baselines else []) \
                 + (["cv"] if cfg.run_cv else []) + ["compare"] + (["edge"] if cfg.run_edge else [])
    else:
        stages = [s.strip() for s in args.stage.split(",")]
        bad = set(stages) - set(STAGES)
        if bad:
            raise SystemExit(f"Unknown stage(s) {bad}; choose from {STAGES}")
    return cfg, stages, args.use_cache


def setup_logging(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(cfg.metrics_dir / "run.log", mode="a")], force=True)


# ============================================================================
# results.json helpers (every stage reads, updates its part, writes back)
# ============================================================================
def load_results(cfg: Config) -> dict:
    path = cfg.metrics_dir / "results.json"
    return json.loads(path.read_text()) if path.exists() else {"models": {}}


def store_results(cfg: Config, results: dict) -> None:
    ev.save_results(results, cfg)


def proba_path(cfg: Config, name: str) -> Path:
    return cfg.array_cache_dir / f"proba_{name.replace(' ', '_')}.npy"


def evaluate_one(name: str, predict, X, data, cfg: Config, extra: dict) -> tuple[dict, np.ndarray]:
    proba, timing = ev.timed_predict(predict, X)
    m = ev.compute_metrics(np.asarray(data.y_test), proba, data.class_names, cfg.threshold)
    m.update(timing, **extra)
    ev.save_classification_report(np.asarray(data.y_test), ev.predictions(proba, cfg.threshold),
                                  data.class_names, cfg, name)
    np.save(proba_path(cfg, name), proba.astype(np.float32))
    log.info("%-12s acc=%.4f f1=%.4f benign_recall=%s mcc=%.4f", name, m["accuracy"],
             m.get("f1", m["f1_macro"]), f"{m.get('benign_recall', float('nan')):.4f}", m["mcc"])
    return m, proba


def keras_predict(net):
    return lambda X: predict_proba(net, X)


def load_data(cfg: Config):
    data, meta = load_cache(cfg)
    cfg.synthetic = meta.get("synthetic", False)
    return data, meta


# ============================================================================
# Stage 1: prepare
# ============================================================================
def stage_prepare(cfg: Config, use_cache: bool) -> None:
    if use_cache:
        load_cache(cfg)  # raises with a clear message if missing / mismatched
        log.info("--use_cache: prepared arrays found, CSVs not read")
        return
    if cache_is_valid(cfg) and not cfg.force:
        log.info("prepare: valid cache in %s -> skipped (use --force to rebuild)", cfg.array_cache_dir)
        return
    try:
        df, population, _ = load_dataset(cfg)
        release = ("original 2023 release (46 features)" if df.shape[1] > 42
                   else "2024 re-release MERGED_CSV (39 features)")
    except FileNotFoundError as err:
        if not cfg.smoke_test:
            log.error("%s", err)
            raise SystemExit(1)
        cfg.synthetic = True
        df, population, _ = make_synthetic(cfg.sample_rows, cfg.seed)
        release = "SYNTHETIC (smoke test only)"
    log_ram("after CSV load")
    n_features_raw = df.shape[1] - 1
    build_label_mapping(df["label"], cfg.metrics_dir / "label_mapping.json")
    df_clean, report = clean(df, cfg)
    del df
    gc.collect()
    data = prepare(df_clean, cfg, report)
    check_no_leakage(data)
    log_ram("after split + scale")

    # ---- EDA figures 1-5 need the raw (unscaled) DataFrame: do them now
    df_train = df_clean.iloc[data.train_idx]
    counts = np.bincount(data.y_train, minlength=data.n_classes)
    after = ({data.class_names[k]: int(round(c * data.class_weight[k])) for k, c in enumerate(counts)}
             if data.class_weight else {data.class_names[k]: int(c) for k, c in enumerate(counts)})
    before = dict(data.train_counts_before_balance)
    if cfg.task != "binary":  # Fig 1 is always a benign-vs-malicious view
        b0 = before.get(data.class_names[0], 0)
        before = {"Benign": b0, "Malicious": sum(before.values()) - b0}
        a0 = after.get(data.class_names[0], 0)
        after = {"Benign": a0, "Malicious": sum(after.values()) - a0}
    ev.plot_class_distribution(population, df_clean["label"], before, after, cfg)
    ev.plot_top_attacks(population, cfg)
    ev.plot_correlation(df_train, data.feature_names, cfg)
    ev.plot_feature_boxplots(df_train, data.feature_names, cfg)
    ev.plot_missing_inf(report, cfg)
    del df_train

    # ---- raw (unscaled) stratified train+val subset for leakage-free CV
    trval = np.concatenate([data.train_idx, data.val_idx])
    y_trval, _ = encode_target(df_clean["label"].iloc[trval], cfg.task)
    pick = trval[stratified_subset(y_trval, cfg.cv_max_rows, cfg.seed)]
    cv_X = df_clean[data.feature_names].to_numpy(np.float32)[pick]
    cv_y, _ = encode_target(df_clean["label"].iloc[pick], cfg.task)

    meta = {"synthetic": cfg.synthetic, "dataset_release": release,
            "population_counts": population.to_dict(), "n_features_raw": n_features_raw}
    save_cache(data, cfg, meta, cv_X, cv_y)

    # New arrays invalidate every downstream artefact (checkpoints, results).
    stale = [*cfg.model_dir.glob("*_last.keras"), *cfg.model_dir.glob("*_state.json"),
             *cfg.array_cache_dir.glob("proba_*.npy"), cfg.metrics_dir / "results.json",
             cfg.metrics_dir / "cv_results.csv", cfg.metrics_dir / "edge_benchmark.csv",
             *cfg.metrics_dir.glob("history_*.csv")]
    for f in stale:
        f.unlink(missing_ok=True)
    results = {"models": {}}
    results["run_info"] = {
        "synthetic": cfg.synthetic, "dataset_release": release, "data_dir": str(cfg.data_dir),
        "task": cfg.task, "seed": cfg.seed, "sample_rows_requested": cfg.sample_rows,
        "population_rows": int(population.sum()), "rows_sampled": report["rows_in"],
        "rows_after_cleaning": report["rows_out"],
        "duplicates_removed": report.get("duplicates_removed", 0),
        "n_features_raw": n_features_raw, "n_features_used": len(data.feature_names),
        "dropped_constant": data.dropped_constant, "balance": cfg.balance, "scaler": cfg.scaler,
        "split": f"{1 - cfg.val_size - cfg.test_size:.0%}/{cfg.val_size:.0%}/{cfg.test_size:.0%} "
                 "train/val/test (stratified)",
        "n_train": int(len(data.y_train)), "n_val": int(len(data.y_val)),
        "n_test": int(len(data.y_test)), "class_weight": data.class_weight,
        "cv_rows": int(len(cv_y)),
    }
    store_results(cfg, results)
    del df_clean, data, cv_X
    gc.collect()


# ============================================================================
# Stage 2: main CNN-LSTM
# ============================================================================
def stage_train(cfg: Config, device: str) -> None:
    results = load_results(cfg)
    if "CNN-LSTM" in results["models"] and not cfg.force:
        log.info("train: CNN-LSTM results exist -> skipped (use --force to retrain)")
        return
    if cfg.force:
        cfg.resume = False
    data, _ = load_data(cfg)
    n_feat = data.X_train.shape[1]
    set_seeds(cfg)
    model = build_main_model(n_feat, data.n_classes, cfg)
    params = save_summary(model, cfg, "CNN-LSTM")
    log.info("CNN-LSTM params: %s", params)
    model, hist, info = fit_keras(model, data, cfg, "CNN-LSTM")
    model.save(cfg.model_dir / "CNN-LSTM.keras")
    m, proba = evaluate_one("CNN-LSTM", keras_predict(model), data.X_test, data, cfg,
                            {**params, **info, "model_size_mb":
                             (cfg.model_dir / "CNN-LSTM.keras").stat().st_size / 1e6})
    results["models"]["CNN-LSTM"] = m
    y_test = np.asarray(data.y_test)

    # majority-class reference: what "always malicious" scores on the same test split
    maj = ev.compute_metrics(y_test, ev.majority_proba(len(y_test), data.n_classes),
                             data.class_names, cfg.threshold)
    results["models"][ev.MAJORITY] = maj

    # validation-split report (the paper's classification report used validation data)
    val_proba = predict_proba(model, data.X_val)
    results["CNN-LSTM_validation"] = ev.compute_metrics(np.asarray(data.y_val), val_proba,
                                                        data.class_names, cfg.threshold)
    ev.save_classification_report(np.asarray(data.y_val), ev.predictions(val_proba, cfg.threshold),
                                  data.class_names, cfg, "CNN-LSTM_validation")
    results["run_info"].update({"device": device, "architecture": cfg.architecture,
                                "epochs_max": cfg.epochs, "batch_size": cfg.batch_size,
                                "learning_rate": cfg.learning_rate, **info})
    store_results(cfg, results)  # save before the (slower) figures

    pred = ev.predictions(proba, cfg.threshold)
    for slide in (False, True):
        ev.plot_history(hist, "accuracy", cfg, slide)
        ev.plot_history(hist, "loss", cfg, slide)
        ev.plot_confusion(y_test, pred, data.class_names, cfg, normalize=False, slide=slide)
        ev.plot_metric_bars(m, cfg, slide)
    ev.plot_lr(hist, cfg)
    ev.plot_confusion(y_test, pred, data.class_names, cfg, normalize=True)
    if data.n_classes == 2:
        ev.plot_proba_hist(y_test, proba[:, 1], cfg.threshold, cfg)
        ev.plot_threshold_sweep(y_test, proba[:, 1], cfg.threshold, cfg)
    else:
        ev.plot_per_class_f1(y_test, pred, data.class_names, cfg)
    err = ev.error_by_subtype(data.sub_test, y_test, pred)
    err.to_csv(cfg.metrics_dir / "error_by_attack_type.csv")
    ev.plot_error_analysis(err, cfg)
    if cfg.run_tsne:
        rng = np.random.default_rng(cfg.seed)
        pick = np.sort(rng.choice(len(y_test), min(cfg.tsne_samples, len(y_test)), replace=False))
        embedder = keras.Model(model.inputs, model.get_layer(EMBEDDING_LAYER).output)
        emb = predict_proba(embedder, np.asarray(data.X_test[pick]))
        ev.plot_tsne(emb, y_test[pick], data.class_names, cfg)
    plot_roc_pr(cfg, data, results)


def plot_roc_pr(cfg: Config, data, results: dict) -> None:
    """ROC / PR for every model whose test probabilities are cached."""
    if data.n_classes != 2:
        return
    y = np.asarray(data.y_test)
    curves = {n: (y, np.load(proba_path(cfg, n))[:, 1]) for n in results["models"]
              if n != ev.MAJORITY and proba_path(cfg, n).exists()}
    ev.plot_roc(curves, cfg)
    ev.plot_roc(curves, cfg, slide=True)
    ev.plot_pr(curves, cfg)


# ============================================================================
# Stage 3: baselines
# ============================================================================
def stage_baselines(cfg: Config) -> None:
    results = load_results(cfg)
    data, _ = load_data(cfg)
    n_feat = data.X_train.shape[1]
    for name, builder in BASELINE_BUILDERS.items():
        if name in results["models"] and not cfg.force:
            log.info("baseline %s done -> skipped", name)
            continue
        set_seeds(cfg)
        net = builder(n_feat, data.n_classes, cfg)
        p = save_summary(net, cfg, name)
        net, _, info = fit_keras(net, data, cfg, name)
        net.save(cfg.model_dir / f"{name}.keras")
        m, _ = evaluate_one(name, keras_predict(net), data.X_test, data, cfg,
                            {**p, **info, "model_size_mb":
                             (cfg.model_dir / f"{name}.keras").stat().st_size / 1e6})
        results["models"][name] = m
        store_results(cfg, results)
        del net
        keras.backend.clear_session()
        gc.collect()
        log_ram(f"after baseline {name}")
    if "RandomForest" not in results["models"] or cfg.force:
        rf, t, n_rows = fit_random_forest(data, cfg)
        rf_path = cfg.model_dir / "RandomForest.joblib"
        joblib.dump(rf, rf_path, compress=3)
        m, _ = evaluate_one("RandomForest", rf.predict_proba, np.asarray(data.X_test), data, cfg,
                            {"train_time_s": t, "train_rows": n_rows,
                             "model_size_mb": rf_path.stat().st_size / 1e6,
                             "total_params": int(sum(e.tree_.node_count for e in rf.estimators_))})
        results["models"]["RandomForest"] = m
        store_results(cfg, results)
        del rf
        gc.collect()
    order = [n for n in ["CNN-LSTM", "CNN", "LSTM", "MLP", "RandomForest", ev.MAJORITY]
             if n in results["models"]]
    table = pd.DataFrame(results["models"]).T.loc[order]
    ev.plot_model_comparison(table, cfg)
    ev.plot_model_comparison(table, cfg, slide=True)
    ev.plot_cost_comparison(table, cfg)
    plot_roc_pr(cfg, data, results)


# ============================================================================
# Stage 4-6
# ============================================================================
def stage_cv(cfg: Config) -> None:
    out = cfg.metrics_dir / "cv_results.csv"
    if cfg.force:
        out.unlink(missing_ok=True)
    if out.exists() and len(pd.read_csv(out)) >= cfg.cv_folds:
        log.info("cv: %s complete -> skipped (use --force to redo)", out.name)
        return
    data, _ = load_data(cfg)  # validates the cache signature
    X = np.load(cfg.array_cache_dir / "cv_X_raw.npy", mmap_mode="r")
    y = np.load(cfg.array_cache_dir / "cv_y.npy")
    cv = cross_validate(X, y, data.n_classes, cfg)
    results = load_results(cfg)
    results["cross_validation"] = {
        "folds": cfg.cv_folds, "epochs_per_fold": cfg.cv_epochs, "rows": int(len(y)),
        **{c: {"mean": float(cv[c].mean()), "std": float(cv[c].std())}
           for c in ("accuracy", "precision", "recall", "f1", "benign_recall")}}
    store_results(cfg, results)
    ev.plot_cv(cv, cfg)


def stage_compare(cfg: Config) -> None:
    results = load_results(cfg)
    if "CNN-LSTM" not in results["models"]:
        raise SystemExit("compare: run --stage train first")
    if cfg.task != "binary":
        log.info("compare: paper is binary-only -> skipped for task %s", cfg.task)
        return
    out = cfg.metrics_dir / "paper_comparison.csv"
    if (out.exists() and not cfg.force
            and out.stat().st_mtime > (cfg.model_dir / "CNN-LSTM.keras").stat().st_mtime):
        log.info("compare: %s is newer than the model -> skipped", out.name)
        return
    load_data(cfg)  # sets cfg.synthetic for the watermark
    compare_with_paper.run(results["models"]["CNN-LSTM"], results["run_info"], cfg,
                           results["models"].get(ev.MAJORITY))


def stage_edge(cfg: Config) -> None:
    import edge_deploy
    out = cfg.metrics_dir / "edge_benchmark.csv"
    if out.exists() and not cfg.force:
        log.info("edge: %s exists -> skipped", out.name)
        return
    data, _ = load_data(cfg)
    model = keras.models.load_model(cfg.model_dir / "CNN-LSTM.keras")
    results = load_results(cfg)
    results["edge"] = edge_deploy.run(model, data.X_test, np.asarray(data.y_test), cfg).to_dict("records")
    store_results(cfg, results)


# ============================================================================
def main(argv=None) -> dict:
    args = parse_args(argv)
    cfg, stages, use_cache = build_config(args)
    setup_logging(cfg)
    log.info("Stages: %s | output: %s", stages, cfg.output_dir)
    device = setup_gpu(cfg)
    set_seeds(cfg)
    for name in stages:
        with stage(name, cfg.metrics_dir):
            if name == "prepare":
                stage_prepare(cfg, use_cache)
            elif name == "train":
                stage_train(cfg, device)
            elif name == "baselines":
                stage_baselines(cfg)
            elif name == "cv":
                stage_cv(cfg)
            elif name == "compare":
                stage_compare(cfg)
            elif name == "edge":
                stage_edge(cfg)
        gc.collect()
    if cfg.synthetic:
        log.warning("*** SYNTHETIC DATA: numbers above are NOT real results ***")
    log.info("Done. Figures: %s | metrics: %s", cfg.fig_dir, cfg.metrics_dir)
    return load_results(cfg)


if __name__ == "__main__":
    main()
