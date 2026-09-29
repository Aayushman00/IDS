"""Training: GPU setup, seeding, memory-lean data feeding, resumable Keras fit,
Random Forest and leakage-free k-fold CV."""
from __future__ import annotations

import gc
import io
import json
import logging
import math
import os
import time
from contextlib import redirect_stdout
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from config import Config
from model import build_main_model, build_random_forest
from preprocessing import Prepared, make_scaler
from runtime import log_ram, ram_status

log = logging.getLogger(__name__)


def setup_gpu(cfg: Config) -> str:
    """Memory growth: TF takes GPU memory as needed instead of reserving it all."""
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:  # already initialised (e.g. notebook re-run)
            pass
    if cfg.mixed_precision and gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
    device = gpus[0].name if gpus else "CPU"
    log.info("TensorFlow %s | device: %s | mixed precision: %s",
             tf.__version__, device, cfg.mixed_precision and bool(gpus))
    if not gpus:
        log.warning("No GPU visible. On Windows use WSL2 + tensorflow[and-cuda].")
    return device


def set_seeds(cfg: Config) -> None:
    """Seed Python, NumPy and TF. cuDNN kernels may still be non-deterministic
    unless --deterministic is passed (which is slower)."""
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    keras.utils.set_random_seed(cfg.seed)
    if cfg.deterministic_ops:
        tf.config.experimental.enable_op_determinism()


# --------------------------------------------------------------------------
# Data feeding: batches are gathered from (memory-mapped) arrays on the fly,
# so the training set is never duplicated into a second in-memory tensor.
# --------------------------------------------------------------------------
class ArraySequence(keras.utils.PyDataset):
    """(X[idx, :, None], y[idx][, sample_weight]) batches from a 2-D array.

    4 prefetch threads keep the GPU fed; indices are reshuffled every epoch
    with a seeded RNG.
    """

    def __init__(self, X, y, batch_size: int, shuffle: bool, seed: int,
                 class_weight: dict | None = None):
        super().__init__(workers=4, use_multiprocessing=False, max_queue_size=16)
        self.X, self.y, self.bs, self.shuffle = X, y, batch_size, shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(y))
        if shuffle:
            self.rng.shuffle(self.order)
        self.w = (np.array([class_weight[k] for k in sorted(class_weight)], np.float32)
                  if class_weight else None)

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.bs)

    def __getitem__(self, i: int):
        idx = np.sort(self.order[i * self.bs:(i + 1) * self.bs])  # sorted = sequential reads
        x = np.asarray(self.X[idx], dtype=np.float32)[..., None]
        y = np.asarray(self.y[idx])
        return (x, y, self.w[y]) if self.w is not None else (x, y)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.order)


def predict_proba(model: keras.Model, X, batch_size: int = 4096) -> np.ndarray:
    """Chunked prediction on a 2-D (possibly memory-mapped) array."""
    return model.predict(ArraySequence(X, np.zeros(len(X), np.int32), batch_size, False, 0),
                         verbose=0)


# --------------------------------------------------------------------------
# Callbacks
# --------------------------------------------------------------------------
class LRLogger(keras.callbacks.Callback):
    """Put the current learning rate into the epoch logs (for the LR plot)."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))


class RAMLogger(keras.callbacks.Callback):
    def __init__(self, every: int = 1):
        super().__init__()
        self.every = every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every == 0:
            log.info("[RAM] epoch %d: %s", epoch + 1, ram_status())


class ResumableEarlyStopping(keras.callbacks.EarlyStopping):
    """EarlyStopping whose best/wait counters survive a restart."""

    def __init__(self, resume_state: dict | None = None, **kw):
        super().__init__(**kw)
        self.resume_state = resume_state

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.resume_state:
            self.best = self.resume_state["best"]
            self.wait = self.resume_state["es_wait"]
            self.best_epoch = self.resume_state["best_epoch"]
            self.best_weights = self.resume_state.get("best_weights")


class ResumableReduceLR(keras.callbacks.ReduceLROnPlateau):
    def __init__(self, resume_state: dict | None = None, **kw):
        super().__init__(**kw)
        self.resume_state = resume_state

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        if self.resume_state:
            self.best = self.resume_state["best"]
            self.wait = self.resume_state["rlr_wait"]
            self.cooldown_counter = self.resume_state["rlr_cooldown"]


class EpochCheckpoint(keras.callbacks.Callback):
    """After every epoch: full model (+optimizer state) and a small JSON state."""

    def __init__(self, last_path: Path, state_path: Path, es, rlr, time_offset: float):
        super().__init__()
        self.last_path, self.state_path, self.es, self.rlr = last_path, state_path, es, rlr
        self.t0 = time.perf_counter() - time_offset

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.last_path)
        state = {"epochs_done": epoch + 1, "best": float(self.es.best),
                 "best_epoch": int(self.es.best_epoch), "es_wait": int(self.es.wait),
                 "rlr_wait": int(self.rlr.wait), "rlr_cooldown": int(self.rlr.cooldown_counter),
                 "train_time_s": time.perf_counter() - self.t0}
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.state_path)  # atomic: a crash never leaves half a file


def compile_model(model: keras.Model, cfg: Config) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_summary(model: keras.Model, cfg: Config, name: str) -> dict:
    """Write model.summary() to disk and return parameter counts."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        model.summary(line_length=110)
    trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    total = int(model.count_params())
    text = buf.getvalue() + f"\nTotal params: {total:,}\nTrainable params: {trainable:,}\n"
    fname = "model_summary.txt" if name == "CNN-LSTM" else f"model_summary_{name}.txt"
    (cfg.metrics_dir / fname).write_text(text, encoding="utf-8")
    return {"total_params": total, "trainable_params": trainable}


def fit_keras(model: keras.Model, data: Prepared, cfg: Config, name: str,
              epochs: int | None = None) -> tuple[keras.Model, pd.DataFrame, dict]:
    """Train with EarlyStopping / ReduceLROnPlateau / best-checkpoint / CSVLogger.

    Resumable: if ``{name}_last.keras`` + ``{name}_state.json`` exist and
    cfg.resume is set, training continues from the last completed epoch with
    the optimizer state, LR and early-stopping counters restored.
    Returns the BEST model (lowest val_loss), full history and run info.
    """
    epochs = epochs or cfg.epochs
    best_path = cfg.model_dir / f"{name}.keras"
    last_path = cfg.model_dir / f"{name}_last.keras"
    state_path = cfg.model_dir / f"{name}_state.json"
    hist_path = cfg.metrics_dir / f"history_{name}.csv"

    state, initial_epoch = None, 0
    if cfg.resume and last_path.exists() and state_path.exists() and hist_path.exists():
        state = json.loads(state_path.read_text())
        initial_epoch = state["epochs_done"]
        model = keras.models.load_model(last_path)  # includes optimizer + current LR
        if best_path.exists():
            state["best_weights"] = keras.models.load_model(best_path).get_weights()
        log.info("RESUMING %s from epoch %d (best val_loss %.5f @ epoch %d)",
                 name, initial_epoch, state["best"], state["best_epoch"] + 1)
    else:
        for p in (last_path, state_path, hist_path):
            p.unlink(missing_ok=True)
        compile_model(model, cfg)

    es = ResumableEarlyStopping(resume_state=state, monitor="val_loss",
                                patience=cfg.early_stop_patience, restore_best_weights=True,
                                verbose=1)
    rlr = ResumableReduceLR(resume_state=state, monitor="val_loss", factor=cfg.reduce_lr_factor,
                            patience=cfg.reduce_lr_patience, min_lr=1e-6, verbose=1)
    callbacks = [
        LRLogger(),  # must precede CSVLogger so 'lr' is written
        es, rlr,
        keras.callbacks.CSVLogger(str(hist_path), append=True),
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=state["best"] if state else None),
        EpochCheckpoint(last_path, state_path, es, rlr,
                        state["train_time_s"] if state else 0.0),
        RAMLogger(every=1),
    ]
    stopped = False
    if initial_epoch < epochs and not (state and state["es_wait"] >= cfg.early_stop_patience):
        train_seq = ArraySequence(data.X_train, data.y_train, cfg.batch_size, True,
                                  cfg.seed + initial_epoch, data.class_weight)
        val_seq = ArraySequence(data.X_val, data.y_val, 4096, False, 0)
        log_ram(f"{name} fit start")
        model.fit(train_seq, validation_data=val_seq, epochs=epochs,
                  initial_epoch=initial_epoch, callbacks=callbacks, verbose=2)
        stopped = es.stopped_epoch > 0
    else:
        log.info("%s already finished training (state: %s)", name, state_path.name)
        stopped = bool(state and state["es_wait"] >= cfg.early_stop_patience)

    final_state = json.loads(state_path.read_text())
    hist = pd.read_csv(hist_path)
    best = keras.models.load_model(best_path)  # evaluate the best-val_loss weights
    info = {"train_time_s": final_state["train_time_s"], "epochs_run": int(len(hist)),
            "best_epoch": int(hist["val_loss"].idxmin()) + 1, "early_stopped": stopped}
    log.info("%s: %d epochs, best epoch %d, %.1f s", name, info["epochs_run"],
             info["best_epoch"], info["train_time_s"])
    return best, hist, info


def stratified_subset(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorted indices of a stratified subsample of size n (all rows if n >= len)."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def fit_random_forest(data: Prepared, cfg: Config):
    """RF on a stratified subsample (default 300k) with capped threads."""
    idx = stratified_subset(np.asarray(data.y_train), cfg.rf_max_rows, cfg.seed)
    rf = build_random_forest(cfg, use_class_weight=data.class_weight is not None)
    rf.set_params(n_jobs=cfg.rf_n_jobs)
    X, y = np.asarray(data.X_train[idx]), np.asarray(data.y_train[idx])
    log.info("RandomForest on %s train rows, n_jobs=%d", f"{len(idx):,}", cfg.rf_n_jobs)
    t0 = time.perf_counter()
    rf.fit(X, y)
    return rf, time.perf_counter() - t0, len(idx)


def cross_validate(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: Config) -> pd.DataFrame:
    """Stratified k-fold on the cached TRAIN+VAL subset (test never touched).

    The scaler is re-fitted inside every fold (leakage-free). Each fold's
    model is deleted and the TF session cleared before the next fold.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    avg = "binary" if n_classes == 2 else "macro"
    out_csv = cfg.metrics_dir / "cv_results.csv"
    records = pd.read_csv(out_csv).to_dict("records") if (out_csv.exists() and cfg.resume) else []
    done = {int(r["fold"]) for r in records}
    skf = StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    for fold, (tr, va) in enumerate(tqdm(list(skf.split(X, y)), desc="CV folds"), 1):
        if fold in done:
            log.info("CV fold %d already done, skipping", fold)
            continue
        scaler = make_scaler(cfg.scaler).fit(X[tr])
        Xtr = scaler.transform(X[tr]).astype(np.float32)
        Xva = scaler.transform(X[va]).astype(np.float32)
        counts = np.bincount(y[tr])
        cw = ({i: len(tr) / (len(counts) * c) for i, c in enumerate(counts)}
              if cfg.balance == "class_weight" else None)
        keras.utils.set_random_seed(cfg.seed + fold)
        model = compile_model(build_main_model(X.shape[1], n_classes, cfg), cfg)
        model.fit(ArraySequence(Xtr, y[tr], cfg.batch_size, True, cfg.seed + fold, cw),
                  epochs=cfg.cv_epochs, verbose=0)
        pred = predict_proba(model, Xva).argmax(1)
        rec = {"fold": fold,
               "accuracy": accuracy_score(y[va], pred),
               "precision": precision_score(y[va], pred, average=avg, zero_division=0),
               "recall": recall_score(y[va], pred, average=avg, zero_division=0),
               "f1": f1_score(y[va], pred, average=avg, zero_division=0),
               "benign_recall": recall_score(y[va], pred, labels=[0], average="macro",
                                             zero_division=0)}
        log.info("CV fold %d: %s | %s", fold, json.dumps({k: round(v, 4) for k, v in rec.items()}),
                 ram_status())
        records.append(rec)
        pd.DataFrame(records).sort_values("fold").to_csv(out_csv, index=False)  # save per fold
        del model, Xtr, Xva, scaler
        keras.backend.clear_session()
        gc.collect()
    return pd.DataFrame(records).sort_values("fold")
