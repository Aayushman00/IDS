"""Measure peak RSS per pipeline stage on a sample (default 200k rows).

A background thread samples process RSS every 20 ms so short-lived peaks
(pd.concat, to_numpy copies, tf.data conversion) are captured.
Usage: python tools/profile_memory.py --sample 200000
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROC = psutil.Process()


class PeakRSS:
    def __init__(self):
        self.peak = 0
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, PROC.memory_info().rss)
            time.sleep(0.02)

    def reset(self):
        self.peak = PROC.memory_info().rss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--legacy", action="store_true", help="profile the pre-fix code path")
    args = ap.parse_args()

    import numpy as np
    from config import Config
    import data_loader
    import preprocessing
    import train
    from model import build_main_model, BASELINE_BUILDERS

    mon = PeakRSS()
    rows = []

    def stage(name, fn):
        gc.collect()
        mon.reset()
        before = PROC.memory_info().rss
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        after = PROC.memory_info().rss
        rows.append({"stage": name, "rss_before_mb": before / 1e6, "peak_mb": mon.peak / 1e6,
                     "rss_after_mb": after / 1e6, "seconds": dt})
        print(f"{name:28s} before {before/1e6:7.0f} MB  peak {mon.peak/1e6:7.0f} MB  "
              f"after {after/1e6:7.0f} MB  {dt:6.1f}s", flush=True)
        return out

    cfg = Config(sample_rows=args.sample, output_dir=Path("/tmp/profile_out"), epochs=1)
    cfg.cache_dir = Path("/tmp/profile_cache")  # never reuse a cached sample
    stage("import tensorflow", lambda: __import__("tensorflow"))
    train.setup_gpu(cfg)
    df, pop, _ = stage("load + sample CSVs", lambda: data_loader.load_dataset(cfg))
    df_clean, rep = stage("clean (dedup/inf)", lambda: preprocessing.clean(df, cfg))
    del df
    data = stage("split+scale (prepare)", lambda: preprocessing.prepare(df_clean, cfg, rep))
    Xtr = stage("reshape to (n,f,1)", lambda: data.as_seq(data.X_train))
    model = build_main_model(data.X_train.shape[1], data.n_classes, cfg)
    stage("train CNN-LSTM 1 epoch", lambda: train.fit_keras(model, data, cfg, "CNN-LSTM", epochs=1))
    stage("predict test", lambda: model.predict(data.as_seq(data.X_test), batch_size=4096, verbose=0))
    stage("RandomForest (n_jobs=-1)", lambda: train.fit_random_forest(data, cfg))
    cfg.cv_epochs, cfg.cv_folds = 1, 2
    stage("CV 2 folds x 1 epoch", lambda: train.cross_validate(df_clean, data, cfg))

    out = Path("/tmp/profile_out") / f"memory_profile_{args.sample}.json"
    out.write_text(json.dumps(rows, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
