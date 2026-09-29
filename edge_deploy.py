"""OPTIONAL: TensorFlow Lite export + single-sample latency benchmark.

Raspberry Pi demo (how this would run on the edge):
  1. copy outputs/models/cnn_lstm_dynamic.tflite + scaler.joblib to the Pi;
  2. `pip install ai-edge-litert joblib numpy` (the slim TFLite runtime);
  3. a flow-feature extractor (e.g. the CICIoT2023 pcap2csv scripts on a
     mirrored switch port, with the ESP32 nodes generating traffic) emits one
     46-value feature vector per window -> scaler.transform -> interpreter
     -> alert when P(malicious) >= threshold.
Latency measured here is on the dev machine's CPU; a Pi 4 is roughly 5-10x slower.
"""
from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from config import Config
from evaluate import MODEL_C, save, style

log = logging.getLogger(__name__)


def _portable_copy(model: keras.Model) -> keras.Model:
    """Same weights, but LSTMs unrolled and forced off cuDNN: when a GPU is visible Keras
    otherwise traces the GPU-only CudnnRNNV3 kernel, which TFLite cannot run."""
    conf = model.get_config()
    for layer in conf["layers"]:
        if layer["class_name"] == "LSTM":
            layer["config"]["use_cudnn"] = False
            layer["config"]["unroll"] = True  # fixed-length loop -> plain TFLite builtins
    copy = keras.Model.from_config(conf)
    copy.set_weights(model.get_weights())
    return copy


def convert(model: keras.Model, out: Path, quantize: bool) -> Path:
    """Keras -> SavedModel -> TFLite (float32 or dynamic-range int8 weights)."""
    with tempfile.TemporaryDirectory() as tmp:
        _portable_copy(model).export(tmp, verbose=False)
        conv = tf.lite.TFLiteConverter.from_saved_model(tmp)
        if quantize:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        # builtins only: the model must run on the slim Pi runtime (no Flex delegate)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        out.write_bytes(conv.convert())
    return out


def run_tflite(path: Path, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Invoke one sample at a time (edge reality). Returns proba, ms/sample."""
    interp = tf.lite.Interpreter(model_path=str(path))
    inp = interp.get_input_details()[0]
    interp.resize_tensor_input(inp["index"], [1, X.shape[1], 1])
    interp.allocate_tensors()
    out_idx = interp.get_output_details()[0]["index"]
    probs = np.empty((len(X), interp.get_output_details()[0]["shape"][-1]), np.float32)
    t0 = time.perf_counter()
    for i in range(len(X)):
        interp.set_tensor(inp["index"], X[i: i + 1, :, None].astype(np.float32))
        interp.invoke()
        probs[i] = interp.get_tensor(out_idx)[0]
    return probs, 1000 * (time.perf_counter() - t0) / len(X)


def run(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray, cfg: Config) -> pd.DataFrame:
    n = min(cfg.edge_eval_samples, len(X_test))
    X, y = X_test[:n], y_test[:n]
    rows = []
    keras_path = cfg.model_dir / "CNN-LSTM.keras"
    t0 = time.perf_counter()
    for i in range(min(n, 500)):  # Keras single-sample latency reference (slow, so capped)
        model(X[i: i + 1, :, None], training=False)
    keras_ms = 1000 * (time.perf_counter() - t0) / min(n, 500)
    keras_acc = float((model.predict(np.asarray(X)[..., None], batch_size=4096, verbose=0).argmax(1)
                       == y).mean())
    rows.append({"variant": "Keras float32 (GPU)", "size_mb": keras_path.stat().st_size / 1e6,
                 "ms_per_sample": keras_ms, "accuracy": keras_acc})
    for quant, fname, label in ((False, "cnn_lstm_float32.tflite", "TFLite float32"),
                                (True, "cnn_lstm_dynamic.tflite", "TFLite dynamic-range int8")):
        path = convert(model, cfg.model_dir / fname, quant)
        proba, ms = run_tflite(path, X)
        rows.append({"variant": label, "size_mb": path.stat().st_size / 1e6,
                     "ms_per_sample": ms, "accuracy": float((proba.argmax(1) == y).mean())})
    df = pd.DataFrame(rows)
    df["accuracy_drop_pp"] = 100 * (df["accuracy"].iloc[0] - df["accuracy"])
    df.to_csv(cfg.metrics_dir / "edge_benchmark.csv", index=False)
    (cfg.metrics_dir / "edge_benchmark.md").write_text(df.to_markdown(index=False, floatfmt=".4f"))
    log.info("Edge benchmark (%d test samples, batch=1):\n%s", n, df.to_string(index=False))
    plot(df, n, cfg)
    return df


def plot(df: pd.DataFrame, n: int, cfg: Config) -> Path:
    """Fig 23: size, latency and accuracy of Keras vs TFLite variants."""
    colors = [MODEL_C["CNN-LSTM"], "#2A7AB0", "#3C9D5D"]
    with style(False):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, col, lab, fmt in zip(axes, ("size_mb", "ms_per_sample", "accuracy"),
                                     ("Model size (MB)", "Latency (ms / sample, batch 1, CPU)",
                                      f"Accuracy on {n:,} test rows"), ("%.3f", "%.3f", "%.4f")):
            b = ax.bar(df["variant"], df[col], color=colors)
            ax.bar_label(b, fmt=fmt)
            ax.set_ylabel(lab)
            ax.tick_params(axis="x", rotation=15)
            if col == "accuracy":
                ax.set_ylim(df[col].min() - 0.01, 1.0)
        fig.suptitle("Fig 23 - Edge deployment: Keras vs TensorFlow Lite")
        return save(fig, "23_edge_tflite_benchmark", cfg)
