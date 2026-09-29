"""Load CICIoT2023 CSV part-files memory-efficiently.

Works with both public CICIoT2023 releases:
  * original 2023 release: ``part-*.csv`` / 46 features + ``label``
  * 2024 re-release (``MERGED_CSV/Merged*.csv``): 39 features + ``Label``
The label column is detected automatically; all other columns are features.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import DOWNLOAD_HELP, Config

log = logging.getLogger(__name__)

LABEL_CANDIDATES = ("label", "Label")


def list_csv_files(data_dir: Path) -> list[Path]:
    """Return all CSV part-files below ``data_dir`` (sorted, cache excluded)."""
    files = sorted(p for p in Path(data_dir).rglob("*.csv") if "cache" not in p.parts)
    if not files:
        raise FileNotFoundError(DOWNLOAD_HELP.format(data_dir=Path(data_dir).resolve()))
    return files


def detect_label_column(csv_path: Path) -> str:
    """Find the label column name from the CSV header."""
    header = pd.read_csv(csv_path, nrows=0).columns
    for name in LABEL_CANDIDATES:
        if name in header:
            return name
    raise ValueError(f"No label column {LABEL_CANDIDATES} in {csv_path}")


def count_rows(files: list[Path], cache_file: Path) -> int:
    """Count data rows by counting newlines (much faster than parsing CSV).

    Cached in JSON keyed by file name + size, because a 10 GB scan is slow.
    """
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    total = 0
    for f in tqdm(files, desc="Counting rows", unit="file"):
        key = f"{f.name}:{f.stat().st_size}"
        if key not in cache:
            n = 0
            with open(f, "rb") as fh:
                while block := fh.read(1 << 24):
                    n += block.count(b"\n")
            cache[key] = n - 1  # minus header
        total += cache[key]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache))
    return total


def _chain(first, rest):
    yield first
    yield from rest


def _downcast(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """float64 -> float32, int64 -> int32: halves RAM, no accuracy impact."""
    for col in df.columns:
        if col == label_col:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_dataset(cfg: Config) -> tuple[pd.DataFrame, pd.Series, str]:
    """Read (a sample of) all CSVs.

    Sampling: each row is kept independently with p = sample_rows / total_rows
    (per class, this is a stratified sample in expectation: class ratios of
    the full 46M-row dataset are preserved without needing a second pass).

    Returns:
        df: sampled rows (features + label column, label column renamed ``label``)
        population_counts: label counts over *all* rows read (before sampling)
        label_col: original label column name
    """
    files = list_csv_files(cfg.data_dir)
    if cfg.max_files:
        files = files[: cfg.max_files]
    label_col = detect_label_column(files[0])
    log.info("Found %d CSV files, label column '%s'", len(files), label_col)

    cache_path = cfg.cache_dir / (
        f"sample_{cfg.sample_rows}_{len(files)}f_seed{cfg.seed}.pkl"
    )
    if cache_path.exists():
        log.info("Loading cached sample %s", cache_path.name)
        df, population = pd.read_pickle(cache_path)
        return df, population, label_col

    if cfg.sample_rows is None:
        frac = 1.0
    else:
        total = count_rows(files, cfg.cache_dir / "row_counts.json")
        frac = min(1.0, cfg.sample_rows / total)
        log.info("Total rows %s -> keep fraction %.5f", f"{total:,}", frac)

    rng = np.random.default_rng(cfg.seed)
    parts: list[pd.DataFrame] = []
    population = pd.Series(dtype="int64")
    for f in tqdm(files, desc="Reading CSVs", unit="file"):
        # Parse features straight into float32: halves the per-chunk parse buffer
        # compared with pandas' float64 default. Fall back if a file has junk text.
        header = pd.read_csv(f, nrows=0).columns
        dtypes = {c: np.float32 for c in header if c != label_col}
        try:
            reader = pd.read_csv(f, chunksize=cfg.chunk_size, dtype=dtypes)
            first = next(reader)
        except ValueError:
            log.warning("%s has non-numeric feature values -> slow path", f.name)
            reader = pd.read_csv(f, chunksize=cfg.chunk_size, low_memory=False)
            first = next(reader)
        for chunk in _chain(first, reader):
            population = population.add(chunk[label_col].value_counts(), fill_value=0)
            if frac < 1.0:
                # sample inside the chunk -> the concatenated frame never exceeds
                # ~sample_rows; same keep-probability for every class = stratified
                # in expectation (checked in the log against population ratios)
                chunk = chunk[rng.random(len(chunk)) < frac]
            parts.append(_downcast(chunk, label_col))
    df = pd.concat(parts, ignore_index=True).rename(columns={label_col: "label"})
    del parts
    population = population.astype("int64").sort_values(ascending=False)
    ratio = (df["label"].value_counts() / len(df)).reindex(population.index).fillna(0)
    drift = (ratio - population / population.sum()).abs().max()
    log.info("Stratification check: max |sample share - population share| = %.5f", drift)
    log.info("Loaded %s rows x %d columns (%.1f MB)", f"{len(df):,}", df.shape[1],
             df.memory_usage(deep=True).sum() / 1e6)

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    pd.to_pickle((df, population), cache_path)
    return df, population, label_col


# --------------------------------------------------------------------------
# Synthetic fallback -- ONLY for --smoke_test when no real CSVs are present.
# --------------------------------------------------------------------------
SYNTHETIC_FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate",
    "Drate", "fin_flag_number", "syn_flag_number", "rst_flag_number",
    "psh_flag_number", "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count", "HTTP",
    "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP", "ARP",
    "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
SYNTHETIC_LABELS = {
    "DDoS-ICMP_Flood": 0.35, "DDoS-UDP_Flood": 0.2, "DoS-UDP_Flood": 0.1,
    "Mirai-greeth_flood": 0.08, "Recon-PortScan": 0.05, "DNS_Spoofing": 0.04,
    "SqlInjection": 0.02, "DictionaryBruteForce": 0.02, "BenignTraffic": 0.14,
}


def make_synthetic(n_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series, str]:
    """Random data with the original CICIoT2023 schema. NOT real traffic."""
    log.warning("*** USING SYNTHETIC DATA -- results are meaningless, smoke test only ***")
    rng = np.random.default_rng(seed)
    labels = rng.choice(list(SYNTHETIC_LABELS), size=n_rows, p=list(SYNTHETIC_LABELS.values()))
    shift = np.array([hash(lbl) % 7 for lbl in labels], dtype=np.float32)[:, None]
    X = rng.lognormal(0, 1, (n_rows, len(SYNTHETIC_FEATURES))).astype(np.float32) + shift
    df = pd.DataFrame(X, columns=SYNTHETIC_FEATURES)
    df["label"] = labels
    return df, df["label"].value_counts(), "label"
