| metric                                 |    paper |   ours_cnn_lstm |   majority_baseline |   abs_diff_vs_paper |   rel_diff_vs_paper_% |
|:---------------------------------------|---------:|----------------:|--------------------:|--------------------:|----------------------:|
| Accuracy (%)                           |  98.42   |         99.3044 |             97.342  |              0.8844 |                 0.899 |
| Precision, support-weighted (%)        | n/a      |         99.3622 |             94.7546 |            n/a      |               n/a     |
| Recall, support-weighted (%)           | n/a      |         99.3044 |             97.342  |            n/a      |               n/a     |
| F1, support-weighted (%)               |  98.57   |         99.3246 |             96.0308 |              0.7546 |                 0.766 |
| Precision, malicious = positive (%)    | n/a      |         99.8161 |             97.342  |            n/a      |               n/a     |
| Recall / detection rate, malicious (%) | n/a      |         99.4686 |            100      |            n/a      |               n/a     |
| F1, malicious (%)                      | n/a      |         99.6421 |             98.6531 |            n/a      |               n/a     |
| Benign precision (%)                   | n/a      |         82.7405 |              0      |            n/a      |               n/a     |
| Benign recall = specificity (%)        | n/a      |         93.2883 |              0      |            n/a      |               n/a     |
| Benign F1 (%)                          | n/a      |         87.6984 |              0      |            n/a      |               n/a     |
| False positive rate (%)                |   9.17   |          6.7117 |            100      |             -2.4583 |               -26.808 |
| False negative rate (%)                | n/a      |          0.5314 |              0      |            n/a      |               n/a     |
| Matthews corr. coef.                   | n/a      |          0.8751 |              0      |            n/a      |               n/a     |
| ROC-AUC                                | n/a      |          0.9981 |              0.5    |            n/a      |               n/a     |
| PR-AUC (average precision)             | n/a      |          0.9999 |              0.9734 |            n/a      |               n/a     |
| Cross-entropy loss                     |   0.0275 |          0.0162 |              0.4238 |             -0.0113 |               -41.091 |