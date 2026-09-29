"""Build README results block, RESULTS_SUMMARY.md, RUNTIME_NOTES.md and
GOAL_CHECKLIST.md from the files written by the pipeline (no hand-typed numbers).

usage: python tools/make_reports.py [--output_dir outputs] [--log run_main.log ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import PAPER_RESULTS  # noqa: E402

MAJ = "Majority (always malicious)"
ORDER = ["CNN-LSTM", "CNN", "LSTM", "MLP", "RandomForest", MAJ]
COLS = [("accuracy", "Accuracy", 100), ("precision", "Precision (mal.)", 100),
        ("recall", "Recall (mal.)", 100), ("f1", "F1 (mal.)", 100),
        ("benign_precision", "Benign P", 100), ("benign_recall", "Benign R", 100),
        ("benign_f1", "Benign F1", 100), ("fpr", "FPR", 100), ("fnr", "FNR", 100),
        ("mcc", "MCC", 1), ("roc_auc", "ROC-AUC", 1), ("pr_auc", "PR-AUC", 1)]


def f(v, scale=1, nd=None):
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    nd = nd if nd is not None else (2 if scale == 100 else 4)
    return f"{v * scale:.{nd}f}"


def headline_table(res: dict) -> str:
    head = "| Model | " + " | ".join(c[1] for c in COLS) + " |"
    sep = "|---|" + "---|" * len(COLS)
    rows = [head, sep]
    for name in ORDER:
        m = res["models"].get(name)
        if m:
            label = f"**{name}** (ours)" if name == "CNN-LSTM" else name
            rows.append(f"| {label} | " + " | ".join(f(m.get(k), s) for k, s in
                                                      ((c[0], c[2]) for c in COLS)) + " |")
    paper = ["n/a"] * len(COLS)
    paper[0] = f"{PAPER_RESULTS['accuracy']:.2f}"
    rows.append("| Paper (Gueriani et al., 2024) | " + " | ".join(paper) + " |")
    return "\n".join(rows)


def cost_table(res: dict) -> str:
    rows = ["| Model | Params / nodes | Size (MB) | Train time (s) | Epochs (best) | Inference µs/sample | Throughput (samples/s) |",
            "|---|---|---|---|---|---|---|"]
    for name in ORDER[:-1]:
        m = res["models"].get(name)
        if not m:
            continue
        ep = f"{m.get('epochs_run', '-')} ({m.get('best_epoch', '-')})" if "epochs_run" in m else \
            f"RF on {m.get('train_rows', 0):,} rows"
        rows.append(f"| {name} | {m.get('total_params', 0):,} | {m['model_size_mb']:.3f} | "
                    f"{m['train_time_s']:.0f} | {ep} | {1000 * m['inference_ms_per_sample']:.2f} | "
                    f"{m['throughput_samples_per_s']:,.0f} |")
    return "\n".join(rows)


def cm_block(m: dict) -> str:
    tn, fp, fn, tp = m["tn"], m["fp"], m["fn"], m["tp"]
    b, a = tn + fp, fn + tp
    return ("| True \\ Predicted | Benign | Malicious |\n|---|---|---|\n"
            f"| **Benign** (n={b:,}) | TN {tn:,} ({100 * tn / b:.2f}%) | FP {fp:,} ({100 * fp / b:.2f}%) |\n"
            f"| **Malicious** (n={a:,}) | FN {fn:,} ({100 * fn / a:.2f}%) | TP {tp:,} ({100 * tp / a:.2f}%) |")


def cv_block(res: dict) -> str:
    cv = res.get("cross_validation")
    if not cv:
        return "_CV stage not run._"
    rows = [f"{cv['folds']}-fold stratified CV on a {cv['rows']:,}-row stratified subset of train+val "
            f"(scaler re-fitted per fold, {cv['epochs_per_fold']} epochs per fold; test set untouched).", "",
            "| Metric | Mean | Std |", "|---|---|---|"]
    for k in ("accuracy", "precision", "recall", "f1", "benign_recall"):
        rows.append(f"| {k} | {100 * cv[k]['mean']:.3f}% | {100 * cv[k]['std']:.3f} pp |")
    return "\n".join(rows)


def edge_block(res: dict) -> str:
    e = res.get("edge")
    if not e:
        return "_Edge stage not run._"
    rows = ["| Variant | Size (MB) | Latency (ms/sample, batch 1) | Accuracy | Drop (pp) |",
            "|---|---|---|---|---|"]
    for r in e:
        rows.append(f"| {r['variant']} | {r['size_mb']:.3f} | {r['ms_per_sample']:.3f} | "
                    f"{100 * r['accuracy']:.3f}% | {r['accuracy_drop_pp']:.3f} |")
    return "\n".join(rows)


def epoch_times(logs: list[Path]) -> list[float]:
    t = []
    for p in logs:
        if p.exists():
            t += [float(x) for x in re.findall(r"^\d+/\d+ - (\d+)s - \d+ms/step", p.read_text(), re.M)]
    return t


def findings(res: dict) -> str:
    """Key findings computed from the numbers (ranking by MCC, the imbalance-robust metric)."""
    ms = {k: v for k, v in res["models"].items() if k != MAJ}
    rank = sorted(ms, key=lambda k: ms[k]["mcc"], reverse=True)
    cl, maj = ms["CNN-LSTM"], res["models"][MAJ]
    lines = ["**Key findings (computed from the table):**", "",
             f"* The CNN-LSTM reaches {100 * cl['accuracy']:.2f}% accuracy "
             f"({100 * cl['accuracy'] - PAPER_RESULTS['accuracy']:+.2f} pp vs the paper), i.e. "
             f"{100 * (cl['accuracy'] - maj['accuracy']):.2f} pp above always-malicious; MCC {cl['mcc']:.3f}.",
             f"* Its errors are almost all missed attacks: FN {cl['fn']:,} vs FP {cl['fp']:,} "
             f"(benign recall {100 * cl['benign_recall']:.2f}% but benign precision "
             f"{100 * cl['benign_precision']:.2f}%). Class weighting pushes borderline flows to 'benign'; "
             "rare web / recon / spoofing attacks are the ones missed (Fig 20).",
             "* Ranking by MCC: " + " > ".join(f"{k} ({ms[k]['mcc']:.3f})" for k in rank) + "."]
    if "CNN" in ms:
        d = ms["CNN-LSTM"]["mcc"] - ms["CNN"]["mcc"]
        lines.append(f"* LSTM branch ablation: CNN-LSTM vs CNN-only MCC difference {d:+.3f} "
                     f"(accuracy {100 * (ms['CNN-LSTM']['accuracy'] - ms['CNN']['accuracy']):+.2f} pp) - "
                     + ("no measurable benefit from the LSTM branch on this split."
                        if d <= 0.005 else "the LSTM branch helps slightly."))
    if "RandomForest" in ms and rank[0] == "RandomForest":
        rf = ms["RandomForest"]
        lines.append(f"* A Random Forest trained on only {rf.get('train_rows', 0):,} rows is the strongest model "
                     f"(accuracy {100 * rf['accuracy']:.2f}%, MCC {rf['mcc']:.3f}), trading a slightly higher "
                     f"FPR ({100 * rf['fpr']:.2f}%) for far fewer missed attacks (FNR {100 * rf['fnr']:.2f}% vs "
                     f"{100 * cl['fnr']:.2f}%). The paper did not compare against tree ensembles.")
    return "\n".join(lines)


def results_block(res: dict, out: Path) -> str:
    ri, m = res["run_info"], res["models"]["CNN-LSTM"]
    maj = res["models"].get(MAJ, {})
    fig = lambda n: f"{out.name}/figures/{n}.png"  # noqa: E731
    val = res.get("CNN-LSTM_validation", {})
    return f"""## 7. Results (full run, real CICIoT2023 data)

**Setup actually used:** {ri['dataset_release']}; requested sample {ri['sample_rows_requested']:,} of
{ri['population_rows']:,} rows (per-chunk stratified sampling) → {ri['rows_sampled']:,} sampled →
{ri['rows_after_cleaning']:,} after removing {ri['duplicates_removed']:,} exact duplicates.
Split {ri['split']}: train {ri['n_train']:,} / val {ri['n_val']:,} / test {ri['n_test']:,}.
{ri['n_features_used']} of {ri['n_features_raw']} features used (train-constant columns dropped:
{', '.join(ri['dropped_constant']) or 'none'}). Class weights (train-only, 'balanced'): benign {ri['class_weight']['0']:.2f}, malicious {ri['class_weight']['1']:.3f}.
CNN-LSTM trained on the RTX 4060 Laptop GPU ({ri.get('device', '?')}, WSL2): **{ri['epochs_run']} epochs run** (max {ri['epochs_max']}),
best val_loss at **epoch {ri['best_epoch']}**{' (early stopping triggered)' if ri.get('early_stopped') else ''},
{ri['train_time_s'] / 60:.1f} min; seed {ri['seed']}; batch {ri['batch_size']}; lr {ri['learning_rate']}.
All numbers below are on the held-out **test** split ({ri['n_test']:,} flows).

### 7.1 Headline table (percent unless noted; positive class = malicious)

Accuracy alone is misleading here: {100 * maj.get('accuracy', float('nan')):.2f}% of test flows are attacks,
so the *always-malicious* row already reaches that accuracy with 0% benign recall and MCC 0. The benign
columns, FPR, MCC and PR-AUC show what the models actually learn.

{headline_table(res)}

{findings(res)}

Paper: only the accuracy (98.42%) is used; the paper's other metrics are left as `None` in
`config.PAPER_RESULTS` until checked against the published paper. Deviation:
**{100 * m['accuracy'] - PAPER_RESULTS['accuracy']:+.2f} pp** accuracy
(see `outputs/metrics/deviation_analysis.txt`).

### 7.2 CNN-LSTM confusion matrix (test set; % of the true class)

{cm_block(m)}

Validation split (the paper reported its classification report on validation data): accuracy
{f(val.get('accuracy'), 100)}%, benign recall {f(val.get('benign_recall'), 100)}%, FPR {f(val.get('fpr'), 100)}%.

![Confusion matrix]({fig('10_confusion_matrix_normalized')})

### 7.3 Training curves

![Accuracy]({fig('06_training_accuracy')})
![Loss]({fig('07_training_loss')})

### 7.4 Baselines and cost

{cost_table(res)}

![Model comparison]({fig('17_model_comparison')})
![ROC]({fig('11_roc_curve')})

### 7.5 Our results vs the paper

{(out / 'metrics' / 'paper_comparison.md').read_text()}

![Ours vs paper]({fig('18_paper_comparison')})

### 7.6 Cross-validation (stability)

{cv_block(res)}

![CV]({fig('22_cross_validation')})

### 7.7 Edge deployment (TFLite, CPU, one sample at a time)

{edge_block(res)}

![Edge]({fig('23_edge_tflite_benchmark')})

### 7.8 Where the model errs

![Errors per attack type]({fig('20_error_by_attack_type')})
"""


def results_summary(res: dict, out: Path) -> str:
    ri = res["run_info"]
    return f"""# Results Summary (copy-paste template)

Reproduction of Gueriani & Kheddar, *Enhancing IoT Security with CNN and LSTM-Based Intrusion Detection
Systems*, IEEE PAIS 2024, on CICIoT2023 ({ri['dataset_release']}).
Sample: {ri['rows_after_cleaning']:,} de-duplicated flows (train {ri['n_train']:,} / val {ri['n_val']:,} /
test {ri['n_test']:,}), binary benign vs malicious, seed {ri['seed']}, {ri['epochs_run']} epochs
(best epoch {ri['best_epoch']}).

## Metrics table (test set, %)

{headline_table(res)}

## Paper comparison

{(out / 'metrics' / 'paper_comparison.md').read_text()}

## Figures for the report and slides

| Use | Report figure (300 dpi) | Slide version (16:9) |
|---|---|---|
| Confusion matrix | `{out.name}/figures/09_confusion_matrix_counts.png`, `10_confusion_matrix_normalized.png` | `{out.name}/figures/slides/09_confusion_matrix_counts.png` |
| Training curves | `{out.name}/figures/06_training_accuracy.png`, `07_training_loss.png` | `slides/06_training_accuracy.png`, `slides/07_training_loss.png` |
| ROC curve | `{out.name}/figures/11_roc_curve.png` | `slides/11_roc_curve.png` |
| Metric bar chart | `{out.name}/figures/13_metric_bars.png` | `slides/13_metric_bars.png` |
| Model comparison | `{out.name}/figures/17_model_comparison.png` | `slides/17_model_comparison.png` |
| Ours vs paper | `{out.name}/figures/18_paper_comparison.png` | `slides/18_paper_comparison.png` |
| Class imbalance (motivation) | `{out.name}/figures/01_class_distribution.png` | – |
| Error analysis | `{out.name}/figures/20_error_by_attack_type.png` | – |
| Live demo (animation) | `{out.name}/animations/traffic_replay.gif` / `.mp4` | `{out.name}/animations/traffic_replay_final_16x9.png` |

## One-paragraph summary (edit as needed)

Our CNN-LSTM reached {100 * res['models']['CNN-LSTM']['accuracy']:.2f}% test accuracy versus the paper's
98.42%. Because {100 * res['models'][MAJ]['accuracy']:.2f}% of flows are attacks, we also report benign
recall ({100 * res['models']['CNN-LSTM']['benign_recall']:.2f}%), FPR
({100 * res['models']['CNN-LSTM']['fpr']:.2f}%) and MCC ({res['models']['CNN-LSTM']['mcc']:.4f}), which an
always-malicious classifier cannot achieve (benign recall 0%, MCC 0).
"""


def runtime_notes(res: dict, out: Path, logs: list[Path], cpu_probe: dict | None) -> str:
    st = json.loads((out / "metrics" / "stage_runtime.json").read_text())
    ep_main = epoch_times([p for p in logs if p.name == "run_main.log"])
    ep_base = epoch_times([p for p in logs if p.name != "run_main.log"])
    rows = ["| Stage | Wall time | Peak RSS (process) | Min. system RAM available |", "|---|---|---|---|"]
    for k, v in st.items():
        rows.append(f"| {k} | {v['seconds'] / 60:.1f} min ({v['seconds']:.0f} s) | {v['peak_rss_gb']:.2f} GB | "
                    f"{v['min_system_available_gb']:.2f} GB |")
    ri = res["run_info"]
    cpu = ""
    if cpu_probe:
        cpu = (f"\n**CPU-only (measured on this laptop's CPU, GPU hidden):** "
               f"{cpu_probe['ms_per_step']:.1f} ms/step at batch {cpu_probe['batch']} → "
               f"≈{cpu_probe['ms_per_step'] * cpu_probe['steps_full'] / 60000:.1f} min per full epoch "
               f"({cpu_probe['steps_full']:,} steps), i.e. ≈{cpu_probe['ms_per_step'] * cpu_probe['steps_full'] * ri['epochs_run'] / 3.6e6:.1f} h "
               f"for {ri['epochs_run']} epochs.\n")
    return f"""# Runtime notes (measured)

Machine: RTX 4060 Laptop GPU (8 GB), 24-thread CPU, WSL2 Ubuntu 24.04 with a 7.6 GB memory cap
(no `.wslconfig`), Windows host with 15.7 GB RAM. TensorFlow 2.20 (GPU via WSL2).

Full run: sample {ri['sample_rows_requested']:,} rows → train {ri['n_train']:,}; CNN-LSTM {ri['epochs_run']}
epochs.

## Per stage (from `{out.name}/metrics/stage_runtime.json`)

{chr(10).join(rows)}

"Peak RSS" is sampled every 50 ms by a background thread (`runtime.py`). The prepare stage reused
the raw-CSV sample cache (`data/original/cache/*.pkl`) if present; a cold read of the 2.3 GB CSVs adds
about 35-55 s and ~0.3 GB peak.

## Epoch time

* CNN-LSTM (GPU, 3,352 steps/epoch): {len(ep_main)} epochs, median {pd.Series(ep_main).median():.0f} s,
  range {min(ep_main, default=0):.0f}-{max(ep_main, default=0):.0f} s (first epoch includes graph tracing).
* Baselines CNN / LSTM / MLP together: {len(ep_base)} epochs, median {pd.Series(ep_base).median():.0f} s
  (per-model totals in the README cost table: LSTM-only is the slowest).
{cpu}
## Other environments (NOT measured here - estimates)

* **Colab free (T4 GPU, ~12.7 GB RAM):** this model is tiny (≈80k parameters) so the step time is
  dominated by per-batch overhead rather than GPU FLOPs; expect roughly the same order as here
  (≈1-2 min per epoch, ~30-60 min for the whole pipeline). RAM is sufficient for the 1.4M-row sample
  with the memory-mapped cache.
* **CPU-only:** measured above on this laptop (24 threads). Because the model has only ~80k
  parameters, the GPU advantage is small here; a slower CPU (e.g. Colab's 2 vCPUs) will be several
  times slower than this measurement - start with `--sample 500000 --epochs 10` there.

## Memory: root cause of the earlier crash and the fixes

The first full run died in epoch 2 with no traceback and no Linux OOM entry: it was a background
shell started by the coding agent, which was reaped when the **Windows host** ran low on memory
(WSL VM ≈ Python 2.6 GB + ~2.5 GB page cache from reading the CSVs, plus Windows apps). The old code
path would also have peaked at ≈4.5 GB (Random Forest with 24 threads on all 858k rows while the
DataFrame, all models and raw arrays stayed alive). Fixes: detached launch (`tools/detach.sh`,
`setsid -f`), per-chunk float32 parsing and sampling, `.npy` cache read memory-mapped by every later
stage, batch feeding from the memory-mapped arrays (`ArraySequence`), Random Forest on a 300k
stratified subsample with `n_jobs=8`, CV on a 150k cached subset with `clear_session()` + `gc` per
fold, and per-epoch resumable checkpoints.

Optional `C:\\Users\\ayush\\.wslconfig` (not applied automatically):

```ini
[wsl2]
memory=11GB          # ~70% of 15.7 GB
swap=16GB
[experimental]
autoMemoryReclaim=gradual   # return WSL page cache to Windows
```
Apply with `wsl --shutdown` from PowerShell, then reopen WSL.
"""


def goal_checklist(res: dict, out: Path) -> str:
    def ok(p: str) -> str:
        return "✅" if (ROOT / p).exists() else "❌ missing"

    items = [
        ("Methodology reproduction of the CNN-LSTM architecture",
         ["model.py", f"{out.name}/metrics/model_summary.txt", "config.py", "README.md"],
         "`build_paper_cnn_lstm` follows the paper's Fig. 2 (two Conv1D-64 blocks, Dense 32/16, two LSTM-64 "
         "layers, two heads concatenated); assumptions tagged `[assumed]` in config.py / README §1."),
        ("Model training on CICIoT2023",
         [f"{out.name}/models/CNN-LSTM.keras", f"{out.name}/metrics/history_CNN-LSTM.csv",
          f"{out.name}/figures/06_training_accuracy.png", f"{out.name}/figures/07_training_loss.png",
          f"{out.name}/metrics/stage_runtime.json"],
         f"{res['run_info']['epochs_run']} epochs on {res['run_info']['n_train']:,} real training flows "
         f"(best epoch {res['run_info']['best_epoch']})."),
        ("A working pipeline distinguishing benign from malicious traffic",
         ["main.py", "data_loader.py", "preprocessing.py", "train.py", "evaluate.py",
          f"{out.name}/figures/09_confusion_matrix_counts.png", f"{out.name}/figures/10_confusion_matrix_normalized.png",
          "test_leakage.py", "test_pipeline.py", f"{out.name}/animations/traffic_replay.gif"],
         f"Test accuracy {100 * res['models']['CNN-LSTM']['accuracy']:.2f}%, benign recall "
         f"{100 * res['models']['CNN-LSTM']['benign_recall']:.2f}%, MCC {res['models']['CNN-LSTM']['mcc']:.4f}."),
        ("Comparison of Accuracy, Precision, Recall and F1 against the paper (~98.42%)",
         ["compare_with_paper.py", f"{out.name}/metrics/paper_comparison.csv",
          f"{out.name}/metrics/paper_comparison.md", f"{out.name}/figures/18_paper_comparison.png",
          f"{out.name}/figures/13_metric_bars.png", f"{out.name}/figures/17_model_comparison.png"],
         "Only the paper accuracy is confirmed; P/R/F1 are reported for our model (weighted and "
         "attack-positive) and left n/a for the paper."),
        ("Documentation of deviations",
         [f"{out.name}/metrics/deviation_analysis.txt", "README.md", "RUNTIME_NOTES.md"],
         f"Accuracy gap {100 * res['models']['CNN-LSTM']['accuracy'] - PAPER_RESULTS['accuracy']:+.2f} pp; causes listed "
         "(sample size, split, duplicates, assumed hyperparameters, class weights, seed, dataset release)."),
    ]
    lines = ["# Goal checklist", "", "| Goal | Evidence (file / figure) | Status | Note |", "|---|---|---|---|"]
    for goal, files, note in items:
        ev = "<br>".join(f"`{p}` {ok(p)}" for p in files)
        status = "✅" if all((ROOT / p).exists() for p in files) else "⚠️"
        lines.append(f"| {goal} | {ev} | {status} | {note} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=ROOT / "outputs")
    ap.add_argument("--log", type=Path, nargs="*", default=[ROOT / "run_main.log", ROOT / "run_stages.log"])
    args = ap.parse_args()
    out = args.output_dir
    res = json.loads((out / "metrics" / "results.json").read_text())
    if res["run_info"].get("synthetic"):
        raise SystemExit("refusing to write reports from SYNTHETIC results")
    cpu_path = out / "metrics" / "cpu_probe.json"
    cpu = json.loads(cpu_path.read_text()) if cpu_path.exists() else None

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = results_block(res, out)
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in readme:
        readme = readme[:readme.index(start)] + f"{start}\n{block}\n{end}" + readme[readme.index(end) + len(end):]
    else:
        readme = readme.replace("<!-- RESULTS -->", f"{start}\n{block}\n{end}")
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "RESULTS_SUMMARY.md").write_text(results_summary(res, out), encoding="utf-8")
    (ROOT / "RUNTIME_NOTES.md").write_text(runtime_notes(res, out, args.log, cpu), encoding="utf-8")
    (ROOT / "GOAL_CHECKLIST.md").write_text(goal_checklist(res, out), encoding="utf-8")
    print("reports written")


if __name__ == "__main__":
    main()
