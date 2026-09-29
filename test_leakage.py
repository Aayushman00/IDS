"""Leakage / label-mapping checks. Run: python -m pytest -q test_leakage.py  (or python test_leakage.py)"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from config import Config
from data_loader import make_synthetic
from preprocessing import check_no_leakage, clean, label_family, prepare


def _prepared(balance: str):
    cfg = Config(output_dir=Path(tempfile.mkdtemp()), balance=balance)
    df, _, _ = make_synthetic(3000, seed=0)
    df = df.drop(columns=["Weight"]).assign(Weight=1.0)       # constant column
    df = df.iloc[np.r_[0:3000, 0:200]]                          # 200 exact duplicates
    df_clean, report = clean(df, cfg)
    return prepare(df_clean, cfg, report), report


def test_no_leakage_class_weight():
    p, report = _prepared("class_weight")
    check_no_leakage(p)
    assert report["duplicates_removed"] >= 200
    assert "Weight" in p.dropped_constant
    # scaler statistics equal train-only min/max, NOT the full-data ones
    assert p.scaler.n_samples_seen_ == len(p.train_idx)


def test_undersample_train_only():
    p, _ = _prepared("undersample")
    check_no_leakage(p)
    assert len(set(np.bincount(p.y_train))) == 1                # train balanced
    assert np.bincount(p.y_test)[0] != np.bincount(p.y_test)[1]  # test untouched


def test_label_family_both_releases():
    assert label_family("BenignTraffic") == "Benign" and label_family("BENIGN") == "Benign"
    assert label_family("DDoS-ICMP_Flood") == label_family("DDOS-ICMP_FLOOD") == "DDoS"
    assert label_family("DoS-UDP_Flood") == "DoS"
    assert label_family("MITM-ArpSpoofing") == "Spoofing"
    assert label_family("Backdoor_Malware") == label_family("XSS") == "Web"
    assert label_family("VulnerabilityScan") == "Recon"
    assert label_family("DictionaryBruteForce") == "BruteForce"


if __name__ == "__main__":
    test_no_leakage_class_weight()
    test_undersample_train_only()
    test_label_family_both_releases()
    print("all leakage tests passed")
