"""Generate IoT_IDS_CNN_LSTM.ipynb (run: python tools/make_notebook.py)."""
from pathlib import Path

import nbformat as nbf

C, M = nbf.v4.new_code_cell, nbf.v4.new_markdown_cell


def figs(*names):
    return C("show(" + ", ".join(repr(n) for n in names) + ")")


cells = [
    M("# IoT Network Intrusion Detection — CNN-LSTM IDS on CICIoT2023\n"
      "Reproduction of **Gueriani & Kheddar, _Enhancing IoT Security with CNN and LSTM-Based Intrusion "
      "Detection Systems_, IEEE PAIS 2024** (DOI 10.1109/PAIS62114.2024.10541178).\n\n"
      "Team: Aayushman (231CS105), Ashutosh Kumar (231CS113), Sahil Mengji (231CS151)\n\n"
      "Goal: **reproduce and validate** the paper's ~98.42 % accuracy (not novelty)."),
    M("> **This notebook does not retrain anything.** Training was done by the staged pipeline "
      "(`main.py --stage prepare,train,baselines,cv,compare,edge`, see README). Here we load the cached, "
      "already split and scaled arrays from `outputs/cache/` and the saved model from `outputs/models/`, "
      "re-predict the held-out test set to check that the stored metrics are reproducible, and walk "
      "through every figure and table.\n>\n> To retrain inside the notebook (e.g. on Colab), run the "
      "optional cell in section 1 — it calls the same stages."),
    M("## 0. Setup\nColab: Runtime → Change runtime type → GPU, upload/clone the project folder and `%cd` "
      "into it. Locally on Windows use WSL2 (TensorFlow ≥ 2.11 has no native-Windows GPU build)."),
    C("import json, os, sys\nfrom pathlib import Path\n"
      "import numpy as np, pandas as pd\n"
      "from IPython.display import Image, Markdown, display\n"
      "OUT = Path('outputs')\n"
      "def show(*names):\n"
      "    for n in names:\n"
      "        p = OUT / 'figures' / f'{n}.png'\n"
      "        display(Image(filename=str(p), width=900)) if p.exists() else print('missing:', p)\n"
      "print(os.getcwd())"),
    M("## 1. Optional: (re)build everything\nSkipped by default. Each stage is skipped if its outputs "
      "already exist; `--force` redoes it. Training resumes from the last epoch if interrupted."),
    C("RUN_PIPELINE = False\n"
      "if RUN_PIPELINE:\n"
      "    import main\n"
      "    main.main(['--stage', 'prepare,train,baselines,cv,compare,edge', '--sample', '1400000'])"),
    M("## 2. Configuration and run info\nEvery hyper-parameter is in `config.py`, tagged `[paper]` or "
      "`[assumed]`. Only the paper's accuracy (98.42 %) is used as a reference; other paper metrics are "
      "left as `None` until checked against the published paper."),
    C("from config import Config, PAPER_RESULTS\n"
      "cfg = Config()\n"
      "results = json.loads((OUT / 'metrics' / 'results.json').read_text())\n"
      "print('Paper reference:', {k: v for k, v in PAPER_RESULTS.items() if v is not None})\n"
      "pd.Series(results['run_info']).to_frame('value')"),
    M("## 3. Load cached arrays + saved model and re-check the test metrics"),
    C("import keras\n"
      "from preprocessing import load_cache\n"
      "from train import predict_proba, setup_gpu\n"
      "import evaluate as ev\n"
      "setup_gpu(cfg)\n"
      "cfg.sample_rows = results['run_info']['sample_rows_requested']  # cache signature must match\n"
      "data, meta = load_cache(cfg)          # memory-mapped float32 arrays\n"
      "print({n: getattr(data, n).shape for n in ('X_train', 'X_val', 'X_test')})\n"
      "model = keras.models.load_model(OUT / 'models' / 'CNN-LSTM.keras')\n"
      "proba = predict_proba(model, data.X_test)\n"
      "m = ev.compute_metrics(np.asarray(data.y_test), proba, data.class_names, cfg.threshold)\n"
      "stored = results['models']['CNN-LSTM']\n"
      "for k in ('accuracy', 'f1', 'benign_recall', 'mcc', 'roc_auc'):\n"
      "    print(f'{k:14s} re-computed {m[k]:.6f}   stored {stored[k]:.6f}')\n"
      "    assert abs(m[k] - stored[k]) < 1e-4, k\n"
      "print('Stored metrics reproduced from the saved model.')"),
    M("## 4. Data exploration (Figs 1–5)\n"
      "Leakage-safe preprocessing: exact duplicates dropped *before* splitting; inf→NaN; NaN rows dropped "
      "(<1 %) or train-median imputed; constant columns found on train only; MinMax scaler, imputer and "
      "class weights fitted on train only; stratified 70/15/15 split (`test_leakage.py` asserts this). "
      "About 97 % of traffic is malicious (Fig 1), which is why accuracy alone is misleading."),
    figs("01_class_distribution", "02_top15_attack_types", "03_feature_correlation",
         "04_feature_boxplots", "05_missing_inf_summary"),
    M("## 5. Model — CNN-LSTM as drawn in the paper's Fig. 2\n"
      "Conv1D(64,k3) → BN → AvgPool → Conv1D(64,k3) → AvgPool → Flatten → Dense32 → Dense16, then "
      "(a) Dense(2, softmax) and (b) Reshape(16,1) → LSTM(64) → LSTM(64) → Dense(2, sigmoid), "
      "concatenated → Dense(2, softmax). Pooling size, activations, batch size and learning rate are "
      "assumptions (see README)."),
    C("print((OUT / 'metrics' / 'model_summary.txt').read_text())"),
    M("## 6. Training behaviour (Figs 6–8)"),
    C("pd.read_csv(OUT / 'metrics' / 'history_CNN-LSTM.csv').round(5)"),
    figs("06_training_accuracy", "07_training_loss", "08_learning_rate"),
    M("## 7. Test-set performance (Figs 9–15)\nPositive class = malicious. Because ~97 % of rows are "
      "attacks, look at the **benign** metrics, FPR, MCC and PR-AUC, and compare against the "
      "*always-malicious* baseline row."),
    C("cols = ['accuracy', 'precision', 'recall', 'f1', 'benign_precision', 'benign_recall', 'benign_f1',\n"
      "        'fpr', 'fnr', 'mcc', 'roc_auc', 'pr_auc', 'tp', 'tn', 'fp', 'fn']\n"
      "pd.DataFrame(results['models']).T[cols].loc[['CNN-LSTM', ev.MAJORITY]]"),
    C("print((OUT / 'metrics' / 'classification_report_CNN-LSTM.txt').read_text())"),
    figs("09_confusion_matrix_counts", "10_confusion_matrix_normalized", "11_roc_curve", "12_pr_curve",
         "13_metric_bars", "14_probability_histogram", "15_threshold_sweep"),
    M("## 8. Baselines and cost (Figs 17, 19)\nCNN-only and LSTM-only are ablations of the same "
      "architecture; MLP and Random Forest (300k-row stratified train subsample) are references."),
    C("t = pd.read_csv(OUT / 'metrics' / 'results.csv').set_index('model')\n"
      "t[['accuracy', 'f1', 'benign_recall', 'benign_f1', 'fpr', 'mcc', 'roc_auc', 'pr_auc',\n"
      "   'inference_ms_per_sample', 'model_size_mb', 'train_time_s']].round(5)"),
    figs("17_model_comparison", "19_inference_cost_comparison"),
    M("## 9. Comparison with the paper (Fig 18) and deviation analysis"),
    C("display(Markdown((OUT / 'metrics' / 'paper_comparison.md').read_text()))"),
    figs("18_paper_comparison"),
    C("print((OUT / 'metrics' / 'deviation_analysis.txt').read_text())"),
    M("## 10. Error analysis, embeddings and stability (Figs 20–22)"),
    C("pd.read_csv(OUT / 'metrics' / 'error_by_attack_type.csv').head(15)"),
    figs("20_error_by_attack_type", "21_tsne_lstm_embeddings", "22_cross_validation"),
    C("pd.Series({k: v for k, v in results.get('cross_validation', {}).items()})"),
    M("## 11. Edge deployment (bonus, Fig 23)"),
    C("p = OUT / 'metrics' / 'edge_benchmark.md'\n"
      "display(Markdown(p.read_text())) if p.exists() else print('edge stage not run')"),
    figs("23_edge_tflite_benchmark"),
    M("## 12. Slide-ready figures (16:9)"),
    C("for p in sorted((OUT / 'figures' / 'slides').glob('*.png')):\n"
      "    display(Image(filename=str(p), width=700))"),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "accelerator": "GPU", "colab": {"provenance": []}}
out = Path(__file__).resolve().parents[1] / "IoT_IDS_CNN_LSTM.ipynb"
nbf.write(nb, out)
print("written", out)
