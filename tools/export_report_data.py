"""Collect every number the report/slides need into report/report_data.json.

The docx/pptx builders read ONLY this file, so no number is typed by hand.
usage: python tools/export_report_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PAPER_RESULTS, PAPER_SETUP, TEAM  # noqa: E402


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    m = ROOT / "outputs" / "metrics"
    res = load(m / "results.json")
    if res["run_info"].get("synthetic"):
        raise SystemExit("refusing: synthetic results")
    nob = load(ROOT / "outputs_nobalance" / "metrics" / "results.json")
    if nob is not None and "CNN-LSTM" not in nob.get("models", {}):
        nob = None  # run not finished yet
    data = {
        "team": TEAM,
        "paper": PAPER_RESULTS, "paper_setup": PAPER_SETUP,
        "run_info": res["run_info"], "models": res["models"],
        "validation": res.get("CNN-LSTM_validation"),
        "cv": res.get("cross_validation"), "edge": res.get("edge"),
        "stage_runtime": load(m / "stage_runtime.json"),
        "threshold": load(m / "threshold_tuning.json"),
        "cpu_probe": load(m / "cpu_probe.json"),
        "replay": load(ROOT / "outputs" / "animations" / "replay_metrics.json"),
        "nobalance": None if nob is None else {
            "run_info": nob["run_info"], "model": nob["models"]["CNN-LSTM"],
            "majority": nob["models"].get("Majority (always malicious)"),
            "threshold": load(ROOT / "outputs_nobalance" / "metrics" / "threshold_tuning.json")},
        "error_by_type": [dict(zip(("sub", "error_rate", "errors", "n"), r))
                          for r in (ln.split(",") for ln in
                                    (m / "error_by_attack_type.csv").read_text().splitlines()[1:])],
    }
    out = ROOT / "report" / "report_data.json"
    out.write_text(json.dumps(data, indent=1, default=str))
    print("written", out, "| nobalance:", nob is not None)


if __name__ == "__main__":
    main()
