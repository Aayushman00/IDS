"""Cleaning, label mapping, splitting, scaling, balancing and reshaping.

Leakage rules enforced here (and checked by ``test_leakage.py``):
  * rows are de-duplicated BEFORE splitting, so no identical row sits in
    both train and test;
  * constant-column detection, imputation medians, the scaler and the class
    weights are all fitted on the TRAIN split only;
  * undersampling touches TRAIN only; val/test keep the natural distribution.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import Config

log = logging.getLogger(__name__)

BENIGN = "BENIGN"
FAMILIES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
# Sub-labels whose family cannot be read from a prefix (upper-cased, '-'/'_' removed).
_EXPLICIT_FAMILY = {
    "BENIGNTRAFFIC": "Benign", "BENIGN": "Benign",
    "VULNERABILITYSCAN": "Recon",
    "MITMARPSPOOFING": "Spoofing", "DNSSPOOFING": "Spoofing",
    "DICTIONARYBRUTEFORCE": "BruteForce",
    "SQLINJECTION": "Web", "COMMANDINJECTION": "Web", "XSS": "Web",
    "BACKDOORMALWARE": "Web", "UPLOADINGATTACK": "Web", "BROWSERHIJACKING": "Web",
}
_PREFIX_FAMILY = (("DDOS", "DDoS"), ("DOS", "DoS"), ("MIRAI", "Mirai"), ("RECON", "Recon"))


def label_family(sub_label: str) -> str:
    """Map a CICIoT2023 sub-label (any release / casing) to its 8-class family."""
    key = sub_label.upper().replace("-", "").replace("_", "").replace(" ", "")
    if key in _EXPLICIT_FAMILY:
        return _EXPLICIT_FAMILY[key]
    for prefix, fam in _PREFIX_FAMILY:
        if key.startswith(prefix):
            return fam
    raise ValueError(f"Unknown CICIoT2023 label: {sub_label!r}")


def build_label_mapping(labels: pd.Series, out_path: Path) -> dict:
    """Save {sub_label: {binary, family}} as JSON and return it."""
    mapping = {}
    for lbl in sorted(labels.unique()):
        fam = label_family(lbl)
        mapping[lbl] = {"binary": "Benign" if fam == "Benign" else "Malicious",
                        "binary_id": 0 if fam == "Benign" else 1,
                        "family": fam}
    out_path.write_text(json.dumps(mapping, indent=2))
    return mapping


def encode_target(sub_labels: pd.Series, task: str) -> tuple[np.ndarray, list[str]]:
    """Return integer targets and class names for the chosen task."""
    fam = sub_labels.map(label_family)
    if task == "binary":
        return (fam != "Benign").astype(np.int32).to_numpy(), ["Benign", "Malicious"]
    if task == "multiclass_8":
        names = [f for f in FAMILIES if f in set(fam)]
        return fam.map({n: i for i, n in enumerate(names)}).astype(np.int32).to_numpy(), names
    if task == "multiclass_34":
        # Benign first so class 0 is always benign, rest alphabetical.
        names = sorted(sub_labels.unique(), key=lambda s: (label_family(s) != "Benign", s))
        return sub_labels.map({n: i for i, n in enumerate(names)}).astype(np.int32).to_numpy(), names
    raise ValueError(f"Unknown task {task!r}")


def clean(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Row-level cleaning that is safe before the split (no statistics fitted)."""
    report: dict = {"rows_in": int(len(df))}
    features = [c for c in df.columns if c != "label"]

    # Non-numeric feature columns: booleans -> 0/1, anything else -> coerced.
    for col in features:
        if df[col].dtype == bool:
            df[col] = df[col].astype(np.float32)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            log.warning("Column %s is not numeric -> coercing", col)
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)

    num = df[features]
    report["inf_per_column"] = {c: int(v) for c, v in np.isinf(num).sum().items()}
    df[features] = num.replace([np.inf, -np.inf], np.nan)
    report["nan_per_column"] = {c: int(v) for c, v in df[features].isna().sum().items()}

    bad_rows = df[features].isna().any(axis=1)
    frac_bad = float(bad_rows.mean())
    report["rows_with_nan_or_inf"] = int(bad_rows.sum())
    report["impute_needed"] = frac_bad >= cfg.nan_drop_threshold
    if 0 < frac_bad < cfg.nan_drop_threshold:
        df = df[~bad_rows]
        log.info("Dropped %d rows with NaN/inf (%.4f%%)", bad_rows.sum(), 100 * frac_bad)
    elif report["impute_needed"]:
        log.info("%.2f%% rows have NaN/inf -> median-impute with TRAIN medians", 100 * frac_bad)

    if cfg.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        report["duplicates_removed"] = int(before - len(df))
        log.info("Removed %s exact duplicate rows", f"{before - len(df):,}")
    report["rows_out"] = int(len(df))
    return df.reset_index(drop=True), report


@dataclass
class Prepared:
    """Everything downstream code needs, already split and scaled."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    sub_test: np.ndarray            # fine-grained labels of the test rows
    feature_names: list[str]
    class_names: list[str]
    class_weight: Optional[dict]
    train_idx: np.ndarray           # row ids in the cleaned frame (leakage checks)
    val_idx: np.ndarray
    test_idx: np.ndarray
    scaler: object
    train_counts_before_balance: dict = field(default_factory=dict)
    dropped_constant: list[str] = field(default_factory=list)
    report: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @staticmethod
    def as_seq(X: np.ndarray) -> np.ndarray:
        """(samples, features) -> (samples, features, 1) for Conv1D / LSTM."""
        return X[..., np.newaxis]


def make_scaler(kind: str):
    """MinMax keeps every feature in [0, 1]; Standard is the switchable alternative."""
    return {"minmax": MinMaxScaler(), "standard": StandardScaler()}[kind]


def prepare(df: pd.DataFrame, cfg: Config, clean_report: dict) -> Prepared:
    """Split -> fit-on-train statistics -> transform -> balance."""
    y, class_names = encode_target(df["label"], cfg.task)
    idx = np.arange(len(df))
    # Stratify on the fine label when every class has >= 2 rows per split
    # (keeps rare web attacks in every split); otherwise on the task label.
    counts = df["label"].value_counts()
    strat = df["label"].to_numpy() if counts.min() >= 10 else y
    rel_test = cfg.test_size
    rel_val = cfg.val_size / (1 - cfg.test_size)
    trval_idx, test_idx = train_test_split(idx, test_size=rel_test, stratify=strat,
                                           random_state=cfg.seed)
    train_idx, val_idx = train_test_split(trval_idx, test_size=rel_val,
                                          stratify=strat[trval_idx], random_state=cfg.seed)

    features = [c for c in df.columns if c != "label"]
    X = df[features].to_numpy(dtype=np.float32)
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]

    if clean_report.get("impute_needed"):
        med = np.nanmedian(X_train, axis=0)          # TRAIN statistics only
        for part in (X_train, X_val, X_test):
            r, c = np.where(np.isnan(part))
            part[r, c] = med[c]

    # Constant columns judged on TRAIN only.
    const = X_train.std(axis=0) == 0
    dropped = [f for f, k in zip(features, const) if k]
    if dropped:
        log.info("Dropping %d constant columns: %s", len(dropped), dropped)
    keep = ~const
    X_train, X_val, X_test = X_train[:, keep], X_val[:, keep], X_test[:, keep]
    features = [f for f, k in zip(features, keep) if k]

    scaler = make_scaler(cfg.scaler).fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a).astype(np.float32)
                              for a in (X_train, X_val, X_test))
    joblib.dump(scaler, cfg.model_dir / "scaler.joblib")

    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    train_counts = {class_names[k]: int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}

    if cfg.balance == "undersample":
        X_train, y_train, train_idx = undersample(X_train, y_train, train_idx, cfg.seed)

    class_weight = None
    if cfg.balance == "class_weight":
        classes = np.unique(y_train)
        w = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(v) for c, v in zip(classes, w)}

    for name, arr in (("train", y_train), ("val", y_val), ("test", y_test)):
        u, c = np.unique(arr, return_counts=True)
        log.info("%-5s n=%-9s %s", name, f"{len(arr):,}",
                 ", ".join(f"{class_names[k]}={v:,}" for k, v in zip(u, c)))

    return Prepared(X_train, X_val, X_test, y_train, y_val, y_test,
                    df["label"].to_numpy()[test_idx], features, class_names,
                    class_weight, train_idx, val_idx, test_idx, scaler,
                    train_counts, dropped, clean_report)


def undersample(X: np.ndarray, y: np.ndarray, idx: np.ndarray, seed: int):
    """Randomly shrink every class to the size of the smallest one (TRAIN only)."""
    rng = np.random.default_rng(seed)
    n_min = np.bincount(y).min()
    keep = np.concatenate([rng.choice(np.where(y == c)[0], n_min, replace=False)
                           for c in np.unique(y)])
    keep.sort()
    log.info("Undersampled train %s -> %s rows", f"{len(y):,}", f"{len(keep):,}")
    return X[keep], y[keep], idx[keep]


def check_no_leakage(p: Prepared) -> None:
    """Hard assertions that the split/scaling rules above actually held."""
    s_tr, s_va, s_te = set(p.train_idx), set(p.val_idx), set(p.test_idx)
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "split overlap"
    n_train = sum(p.train_counts_before_balance.values())
    assert int(np.max(p.scaler.n_samples_seen_)) == n_train, "scaler saw non-train rows"
    if isinstance(p.scaler, MinMaxScaler):
        # fitted on train => train lies inside [0, 1]; val/test may exceed it
        assert p.X_train.min() >= -1e-5 and p.X_train.max() <= 1 + 1e-5, "scaler not fit on train"
    if p.class_weight is not None:
        counts = np.bincount(p.y_train)
        expected = len(p.y_train) / (len(counts) * counts)
        got = np.array([p.class_weight[i] for i in range(len(counts))])
        assert np.allclose(got, expected), "class weights not computed from train"
    log.info("Leakage checks passed (disjoint splits, train-only scaler & class weights)")


# --------------------------------------------------------------------------
# Array cache: stage 1 writes it once; every later stage reads only these
# float32 .npy files (memory-mapped), never the raw CSVs or a DataFrame.
# --------------------------------------------------------------------------
_ARRAYS = ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test",
           "train_idx", "val_idx", "test_idx")


def cache_signature(cfg: Config) -> dict:
    """Settings that change the prepared arrays; a mismatch invalidates the cache."""
    return {"data_dir": str(Path(cfg.data_dir).resolve()), "sample_rows": cfg.sample_rows,
            "max_files": cfg.max_files, "task": cfg.task, "seed": cfg.seed,
            "val_size": cfg.val_size, "test_size": cfg.test_size, "scaler": cfg.scaler,
            "balance": cfg.balance, "drop_duplicates": cfg.drop_duplicates,
            "cv_max_rows": cfg.cv_max_rows}


def save_cache(p: Prepared, cfg: Config, meta: dict,
               cv_X: np.ndarray, cv_y: np.ndarray) -> None:
    d = cfg.array_cache_dir
    for name in _ARRAYS:
        arr = getattr(p, name)
        np.save(d / f"{name}.npy", arr.astype(np.float32) if name.startswith("X") else arr)
    names, codes = np.unique(p.sub_test.astype(str), return_inverse=True)
    np.save(d / "sub_test_codes.npy", codes.astype(np.int16))
    np.save(d / "cv_X_raw.npy", cv_X.astype(np.float32))
    np.save(d / "cv_y.npy", cv_y.astype(np.int32))
    meta = {**meta, "signature": cache_signature(cfg), "sub_test_names": names.tolist(),
            "feature_names": p.feature_names, "class_names": p.class_names,
            "class_weight": p.class_weight, "dropped_constant": p.dropped_constant,
            "train_counts_before_balance": p.train_counts_before_balance, "report": p.report}
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log.info("Cached prepared arrays in %s", d)


def cache_is_valid(cfg: Config) -> bool:
    path = cfg.array_cache_dir / "meta.json"
    if not path.exists():
        return False
    return json.loads(path.read_text()).get("signature") == cache_signature(cfg)


def load_cache(cfg: Config) -> tuple[Prepared, dict]:
    """Memory-mapped arrays: pages are read on demand and shared with the OS cache."""
    d = cfg.array_cache_dir
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No prepared cache in {d}. Run: python main.py --stage prepare")
    meta = json.loads(meta_path.read_text())
    if meta.get("signature") != cache_signature(cfg):
        raise ValueError(f"Cache in {d} was built with different settings:\n"
                         f"  cache: {meta.get('signature')}\n  now:   {cache_signature(cfg)}\n"
                         "Re-run `--stage prepare --force` or pass the same options.")
    arr = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in _ARRAYS}
    sub_test = np.asarray(meta["sub_test_names"])[np.load(d / "sub_test_codes.npy")]
    cw = meta["class_weight"]
    p = Prepared(**arr, sub_test=sub_test, feature_names=meta["feature_names"],
                 class_names=meta["class_names"],
                 class_weight={int(k): v for k, v in cw.items()} if cw else None,
                 scaler=joblib.load(cfg.model_dir / "scaler.joblib"),
                 train_counts_before_balance=meta["train_counts_before_balance"],
                 dropped_constant=meta["dropped_constant"], report=meta["report"])
    return p, meta
