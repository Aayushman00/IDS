# Results Summary (copy-paste template)

Reproduction of Gueriani & Kheddar, *Enhancing IoT Security with CNN and LSTM-Based Intrusion Detection
Systems*, IEEE PAIS 2024, on CICIoT2023 (original 2023 release (46 features)).
Sample: 1,225,709 de-duplicated flows (train 857,995 / val 183,857 /
test 183,857), binary benign vs malicious, seed 42, 25 epochs
(best epoch 23).

## Metrics table (test set, %)

| Model | Accuracy | Precision (mal.) | Recall (mal.) | F1 (mal.) | Benign P | Benign R | Benign F1 | FPR | FNR | MCC | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CNN-LSTM** (ours) | 98.74 | 100.00 | 98.71 | 99.35 | 67.96 | 99.84 | 80.87 | 0.16 | 1.29 | 0.8184 | 0.9977 | 0.9999 |
| CNN | 98.82 | 100.00 | 98.79 | 99.39 | 69.24 | 99.84 | 81.77 | 0.16 | 1.21 | 0.8264 | 0.9980 | 0.9999 |
| LSTM | 98.64 | 100.00 | 98.60 | 99.30 | 66.17 | 99.92 | 79.61 | 0.08 | 1.40 | 0.8074 | 0.9969 | 0.9999 |
| MLP | 98.78 | 100.00 | 98.75 | 99.37 | 68.62 | 99.88 | 81.35 | 0.12 | 1.25 | 0.8227 | 0.9980 | 0.9999 |
| RandomForest | 99.61 | 99.95 | 99.65 | 99.80 | 88.55 | 98.08 | 93.07 | 1.92 | 0.35 | 0.9300 | 0.9995 | 1.0000 |
| Majority (always malicious) | 97.34 | 97.34 | 100.00 | 98.65 | 0.00 | 0.00 | 0.00 | 100.00 | 0.00 | 0.0000 | 0.5000 | 0.9734 |
| Paper (Gueriani et al., 2024) | 98.42 | n/a | n/a | n/a | n/a | n/a | n/a | 9.17 | n/a | n/a | n/a | n/a |

## Paper comparison

| metric                                 |    paper |   ours_cnn_lstm |   majority_baseline |   abs_diff_vs_paper |   rel_diff_vs_paper_% |
|:---------------------------------------|---------:|----------------:|--------------------:|--------------------:|----------------------:|
| Accuracy (%)                           |  98.42   |         98.7447 |             97.342  |              0.3247 |                 0.33  |
| Precision, support-weighted (%)        | n/a      |         99.144  |             94.7546 |            n/a      |               n/a     |
| Recall, support-weighted (%)           | n/a      |         98.7447 |             97.342  |            n/a      |               n/a     |
| F1, support-weighted (%)               |  98.57   |         98.8599 |             96.0308 |              0.2899 |                 0.294 |
| Precision, malicious = positive (%)    | n/a      |         99.9955 |             97.342  |            n/a      |               n/a     |
| Recall / detection rate, malicious (%) | n/a      |         98.7149 |            100      |            n/a      |               n/a     |
| F1, malicious (%)                      | n/a      |         99.351  |             98.6531 |            n/a      |               n/a     |
| Benign precision (%)                   | n/a      |         67.9621 |              0      |            n/a      |               n/a     |
| Benign recall = specificity (%)        | n/a      |         99.8363 |              0      |            n/a      |               n/a     |
| Benign F1 (%)                          | n/a      |         80.8719 |              0      |            n/a      |               n/a     |
| False positive rate (%)                |   9.17   |          0.1637 |            100      |             -9.0063 |               -98.215 |
| False negative rate (%)                | n/a      |          1.2851 |              0      |            n/a      |               n/a     |
| Matthews corr. coef.                   | n/a      |          0.8184 |              0      |            n/a      |               n/a     |
| ROC-AUC                                | n/a      |          0.9977 |              0.5    |            n/a      |               n/a     |
| PR-AUC (average precision)             | n/a      |          0.9999 |              0.9734 |            n/a      |               n/a     |
| Cross-entropy loss                     |   0.0275 |          0.0443 |              0.4238 |              0.0168 |                61.091 |

## Figures for the report and slides

| Use | Report figure (300 dpi) | Slide version (16:9) |
|---|---|---|
| Confusion matrix | `outputs/figures/09_confusion_matrix_counts.png`, `10_confusion_matrix_normalized.png` | `outputs/figures/slides/09_confusion_matrix_counts.png` |
| Training curves | `outputs/figures/06_training_accuracy.png`, `07_training_loss.png` | `slides/06_training_accuracy.png`, `slides/07_training_loss.png` |
| ROC curve | `outputs/figures/11_roc_curve.png` | `slides/11_roc_curve.png` |
| Metric bar chart | `outputs/figures/13_metric_bars.png` | `slides/13_metric_bars.png` |
| Model comparison | `outputs/figures/17_model_comparison.png` | `slides/17_model_comparison.png` |
| Ours vs paper | `outputs/figures/18_paper_comparison.png` | `slides/18_paper_comparison.png` |
| Class imbalance (motivation) | `outputs/figures/01_class_distribution.png` | – |
| Error analysis | `outputs/figures/20_error_by_attack_type.png` | – |
| Live demo (animation) | `outputs/animations/traffic_replay.gif` / `.mp4` | `outputs/animations/traffic_replay_final_16x9.png` |

## One-paragraph summary (edit as needed)

Our CNN-LSTM reached 98.74% test accuracy versus the paper's
98.42%. Because 97.34% of flows are attacks, we also report benign
recall (99.84%), FPR
(0.16%) and MCC (0.8184), which an
always-malicious classifier cannot achieve (benign recall 0%, MCC 0).
