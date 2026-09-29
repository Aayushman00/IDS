"""Cache round-trip and training-resume checks. Run: python -m pytest -q test_pipeline.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import Config
from data_loader import make_synthetic
from preprocessing import (check_no_leakage, clean, load_cache, prepare, save_cache)


def _cfg(**kw) -> Config:
    return Config(output_dir=Path(tempfile.mkdtemp()), sample_rows=3000, **kw)


def _prepared(cfg: Config):
    df, _, _ = make_synthetic(3000, seed=0)
    df_clean, report = clean(df, cfg)
    p = prepare(df_clean, cfg, report)
    save_cache(p, cfg, {"synthetic": True}, p.X_train[:100], p.y_train[:100])
    return p


def test_cache_roundtrip_and_leakage():
    cfg = _cfg()
    p = _prepared(cfg)
    q, meta = load_cache(cfg)
    for name in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        assert np.array_equal(np.asarray(getattr(q, name)), getattr(p, name)), name
    assert isinstance(q.X_train, np.memmap)            # memory-mapped, not copied
    assert list(q.sub_test) == list(p.sub_test.astype(str))
    assert q.class_weight == p.class_weight and meta["synthetic"] is True
    check_no_leakage(q)


def test_cache_signature_mismatch_raises():
    cfg = _cfg()
    _prepared(cfg)
    cfg.seed = 7
    with pytest.raises(ValueError, match="different settings"):
        load_cache(cfg)


def test_training_resumes_from_last_epoch():
    """Train 1 epoch, 'crash', resume to 3: history must be continuous, not restarted."""
    from model import build_main_model
    from train import fit_keras

    cfg = _cfg(epochs=1, batch_size=128, early_stop_patience=10)
    p = _prepared(cfg)
    data, _ = load_cache(cfg)
    fit_keras(build_main_model(data.X_train.shape[1], 2, cfg), data, cfg, "T")
    state = json.loads((cfg.model_dir / "T_state.json").read_text())
    assert state["epochs_done"] == 1
    cfg.epochs = 3
    _, hist, info = fit_keras(build_main_model(data.X_train.shape[1], 2, cfg), data, cfg, "T")
    assert list(hist["epoch"]) == [0, 1, 2] and info["epochs_run"] == 3
    assert json.loads((cfg.model_dir / "T_state.json").read_text())["epochs_done"] == 3
