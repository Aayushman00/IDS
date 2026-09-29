"""Single source of truth for paths, seeds and hyperparameters.

Every value that comes from the paper is tagged ``[paper]`` with the place it
was read from (arXiv:2405.18624v1 = the preprint of the IEEE PAIS 2024 paper).
Every value we had to choose ourselves is tagged ``[assumed]`` so it is obvious
what to change if the published version of the paper says otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Numbers reported by Gueriani, Kheddar & Mazari (PAIS 2024).
# CONFIRMED values come from the abstract of the PUBLISHED version
# (DOI 10.1109/PAIS62114.2024.10541178, abstract retrieved via OpenAlex/Crossref
# metadata for that DOI): accuracy, loss, FPR and F1. Precision and recall only
# appear in Table IV of the full text, which we could not access; they stay
# None (the arXiv preprint 2405.18624v1 lists 98.85 / 98.42, unverified against
# the published text). Code that compares with the paper skips None entries.
# --------------------------------------------------------------------------
PAPER_RESULTS: dict[str, Optional[float]] = {
    "accuracy": 98.42,        # [paper] published abstract (also project deck)
    "f1": 98.57,              # [paper] published abstract; averaging not stated (preprint: weighted)
    "fpr": 9.17,              # [paper] published abstract
    "loss": 0.0275,           # [paper] published abstract
    "precision": None,        # TODO: Table IV of the published paper (preprint: 98.85, weighted)
    "recall": None,           # TODO: Table IV of the published paper (preprint: 98.42, weighted)
    "roc_auc": None,          # not reported numerically (only a ROC figure)
    "benign_precision": None,  # TODO: published classification report, if needed
    "benign_recall": None,     # TODO
    "benign_f1": None,         # TODO
}

PAPER_SETUP = {
    "train_val_rows": 1_191_264,   # [paper] Sec. IV-A
    "test_rows": 1_175_692,        # [paper] Sec. IV-A (separate CSV subset)
    "n_features": 45,              # [paper] Sec. III (our CSVs have 46 or 39)
    "train_val_split": "80/20 of subset 1, test = other files",  # [paper] Sec. III
    "epochs": 25,                  # [paper] Sec. IV-C
    "optimizer": "Adam",           # [paper] Sec. IV-C
    "platform": "Google Colab",    # [paper] Sec. IV-C
}

TEAM = "Aayushman (231CS105), Ashutosh Kumar (231CS113), Sahil Mengji (231CS151)"

DOWNLOAD_HELP = """
CICIoT2023 CSV files not found in: {data_dir}

Get them in ONE of these ways, then re-run:
  1. Official (needs a free registration form):
       https://www.unb.ca/cic/datasets/iotdataset-2023.html
     -> download CSV/MERGED_CSV/*.csv (or the original CSV/part-*.csv files)
  2. Public mirror of the official folder (no login), e.g. with curl:
       for i in $(seq -w 1 63); do
         curl -L -o {data_dir}/Merged$i.csv \\
           https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main/CSV/MERGED_CSV/Merged$i.csv
       done
  3. python download_data.py --out {data_dir}

Or run `python main.py --smoke_test` which falls back to a clearly-labelled
SYNTHETIC dataset (only to check that the code runs).
"""


@dataclass
class Config:
    """All run settings. ``main.py`` overrides fields from the CLI."""

    # ---- paths --------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data" / "original"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # ---- reproducibility ----------------------------------------------------
    seed: int = 42
    deterministic_ops: bool = False  # tf op determinism: slower, GPU-exact

    # ---- data ---------------------------------------------------------------
    # None = every row (needs ~10+ GB RAM). Default mirrors the paper's
    # 1,191,264-row train/val subset plus a test share on top of it.
    sample_rows: Optional[int] = 1_400_000   # [assumed] ~paper subset size
    max_files: Optional[int] = None          # read only the first k CSV files
    chunk_size: int = 250_000                # rows per pandas chunk
    task: str = "binary"                     # binary | multiclass_8 | multiclass_34
    val_size: float = 0.15                   # [assumed] 70/15/15 (paper: 80/20 + separate test)
    test_size: float = 0.15
    scaler: str = "minmax"                   # minmax | standard   [assumed]
    balance: str = "class_weight"            # none | class_weight | undersample
    drop_duplicates: bool = True
    nan_drop_threshold: float = 0.01         # drop NaN rows if < 1 %, else impute

    # ---- model --------------------------------------------------------------
    # "paper" = exact topology read from Fig. 2 of the paper.
    # "generic" = the CNN-LSTM default suggested in the project brief.
    architecture: str = "paper"
    conv_filters: int = 64       # [paper] Fig. 2 kernels 3x1x64, 3x64x64
    conv_kernel: int = 3         # [paper] Fig. 2
    pool_size: int = 2           # [assumed] consistent with Flatten=576 for 45 features
    dense_units: tuple[int, int] = (32, 16)   # [paper] Fig. 2 (576x32, 32x16)
    lstm_units: int = 64         # [paper] Fig. 2 recurrent kernel 64x256
    dropout: float = 0.0         # [paper] Fig. 2 shows no dropout layers

    # ---- training -----------------------------------------------------------
    epochs: int = 25             # [paper] Sec. IV-C
    batch_size: int = 256        # [assumed] not stated in the paper
    learning_rate: float = 1e-3  # [assumed] Keras Adam default
    early_stop_patience: int = 5          # [assumed]
    reduce_lr_patience: int = 2           # [assumed]
    reduce_lr_factor: float = 0.5         # [assumed]
    mixed_precision: bool = False         # safe default; LSTM is fine in fp32
    threshold: float = 0.5                # decision threshold for P(malicious)

    # ---- extras -------------------------------------------------------------
    run_baselines: bool = False
    run_cv: bool = False
    cv_folds: int = 5
    cv_epochs: int = 8            # [assumed] shortened so 5 folds stay affordable
    cv_max_rows: int = 150_000    # stratified CV subset of train+val (bounds RAM + time)
    run_tsne: bool = True
    tsne_samples: int = 5000
    run_edge: bool = False
    edge_eval_samples: int = 20_000
    rf_estimators: int = 100
    rf_max_rows: int = 300_000    # stratified train subsample for RF (deep trees are RAM-hungry)
    rf_n_jobs: int = 8            # capped: every worker thread holds its own tree buffers
    resume: bool = True           # continue an interrupted training run from the last epoch
    force: bool = False           # re-run stages whose outputs already exist
    smoke_test: bool = False
    synthetic: bool = False       # set automatically when no real CSVs exist

    # derived paths (filled in __post_init__)
    fig_dir: Path = field(init=False)
    slide_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    array_cache_dir: Path = field(init=False)
    anim_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.refresh_paths()

    def refresh_paths(self) -> None:
        """Recompute derived folders (call after changing output_dir)."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.slide_dir = self.fig_dir / "slides"
        self.metrics_dir = self.output_dir / "metrics"
        self.model_dir = self.output_dir / "models"
        self.cache_dir = self.data_dir / "cache"            # raw-CSV sample + row counts
        self.array_cache_dir = self.output_dir / "cache"    # prepared float32 .npy arrays
        self.anim_dir = self.output_dir / "animations"
        for d in (self.fig_dir, self.slide_dir, self.metrics_dir, self.model_dir,
                  self.array_cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def apply_smoke_test(self) -> None:
        """Tiny, fast settings to prove the whole pipeline runs (<2 min)."""
        self.smoke_test = True
        self.sample_rows = 5_000
        self.max_files = 1
        self.epochs = 2
        self.cv_epochs = 1
        self.cv_max_rows = 3_000
        self.rf_max_rows = 3_000
        self.tsne_samples = 500
        self.edge_eval_samples = 500
        self.rf_estimators = 20
        self.output_dir = PROJECT_ROOT / "outputs_smoke"
        self.refresh_paths()
