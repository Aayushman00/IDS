"""Measure CNN-LSTM training speed with the GPU hidden (CPU-only), on cached data.

Trains 2 short epochs of 150 steps on the cached train arrays and records the
steady-state ms/step (second epoch) -> outputs/metrics/cpu_probe.json.
usage: CUDA_VISIBLE_DEVICES=-1 python tools/cpu_probe.py --sample 1400000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from config import Config  # noqa: E402
from model import build_main_model  # noqa: E402
from preprocessing import load_cache  # noqa: E402
from train import ArraySequence, compile_model  # noqa: E402


class Timer(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.last = time.perf_counter() - self.t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1_400_000)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    assert not tf.config.list_physical_devices("GPU"), "GPU still visible"
    cfg = Config(sample_rows=args.sample)
    data, _ = load_cache(cfg)
    n = args.steps * cfg.batch_size
    seq = ArraySequence(data.X_train[:n], np.asarray(data.y_train[:n]), cfg.batch_size, True, 0,
                        data.class_weight)
    model = compile_model(build_main_model(data.X_train.shape[1], data.n_classes, cfg), cfg)
    t = Timer()
    model.fit(seq, epochs=2, verbose=0, callbacks=[t])
    ms = 1000 * t.last / args.steps
    out = {"device": "CPU only", "cpu_threads": os.cpu_count(), "batch": cfg.batch_size,
           "ms_per_step": ms, "steps_full": int(np.ceil(len(data.y_train) / cfg.batch_size))}
    (cfg.metrics_dir / "cpu_probe.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
