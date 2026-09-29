"""RAM / wall-clock bookkeeping shared by all pipeline stages."""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

log = logging.getLogger(__name__)
_PROC = psutil.Process()


def ram_status() -> str:
    """One-line RAM summary: this process + whole (WSL) system."""
    vm = psutil.virtual_memory()
    return (f"RSS {_PROC.memory_info().rss / 1e9:.2f} GB | system used {vm.used / 1e9:.2f} / "
            f"{vm.total / 1e9:.2f} GB, available {vm.available / 1e9:.2f} GB")


def log_ram(tag: str) -> None:
    log.info("[RAM] %s: %s", tag, ram_status())


class _PeakSampler:
    """Background thread sampling RSS every 50 ms so short peaks are not missed."""

    def __init__(self) -> None:
        self.peak_rss = _PROC.memory_info().rss
        self.min_available = psutil.virtual_memory().available
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.peak_rss = max(self.peak_rss, _PROC.memory_info().rss)
            self.min_available = min(self.min_available, psutil.virtual_memory().available)

    def stop(self) -> None:
        self._stop.set()
        self._t.join()


@contextmanager
def stage(name: str, metrics_dir: Path):
    """Log RAM at start/end and append {seconds, peak RSS} to stage_runtime.json."""
    log.info("=" * 20 + " STAGE %s " + "=" * 20, name)
    log_ram(f"{name} start")
    sampler = _PeakSampler()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sampler.stop()
        dt = time.perf_counter() - t0
        log_ram(f"{name} end")
        log.info("[RAM] %s peak RSS %.2f GB, min system available %.2f GB, %.1f s",
                 name, sampler.peak_rss / 1e9, sampler.min_available / 1e9, dt)
        path = metrics_dir / "stage_runtime.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        data[name] = {"seconds": round(dt, 1), "peak_rss_gb": round(sampler.peak_rss / 1e9, 3),
                      "min_system_available_gb": round(sampler.min_available / 1e9, 3),
                      "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        path.write_text(json.dumps(data, indent=2))
