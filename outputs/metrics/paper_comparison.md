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