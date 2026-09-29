"""Download CICIoT2023 CSVs from public mirrors of the official UNB release.

  original  : 46-feature original release (train/validation/test CSVs, ~2.3 GB)
  merged    : 2024 re-release, 63 MERGED_CSV files, 39 features (~9.5 GB)
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

ORIGINAL_ZIP = "https://huggingface.co/datasets/bhoomig0630/ciciot2023/resolve/main/archive%20(1).zip"
MERGED_URL = ("https://huggingface.co/datasets/bencorn/CIC-IoT-2023/resolve/main/"
              "CSV/MERGED_CSV/Merged{:02d}.csv")


def fetch(url: str, dest: Path) -> None:
    """Download url -> dest unless it already exists."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--release", choices=["original", "merged"], default="original")
    args = ap.parse_args()
    if args.release == "original":
        zip_path = args.out / "original.zip"
        fetch(ORIGINAL_ZIP, zip_path)
        zipfile.ZipFile(zip_path).extractall(args.out / "original")
        zip_path.unlink()
    else:
        for i in tqdm(range(1, 64), desc="Merged CSVs"):
            fetch(MERGED_URL.format(i), args.out / "merged_2024" / f"Merged{i:02d}.csv")


if __name__ == "__main__":
    main()
