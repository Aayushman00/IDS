| metric                                 |    paper |   ours_cnn_lstm |   majority_baseline |   abs_diff_vs_paper |   rel_diff_vs_paper_% |
|:---------------------------------------|---------:|----------------:|--------------------:|--------------------:|----------------------:|
| Accuracy (%)                           |  98.42   |         97.2248 |             96.645  |             -1.1952 |                -1.214 |
| Precision, support-weighted (%)        | n/a      |         98.4738 |             93.4025 |            n/a      |               n/a     |
| Recall, support-weighted (%)           | n/a      |         97.2248 |             96.645  |            n/a      |               n/a     |
| F1, support-weighted (%)               |  98.57   |         97.6092 |             94.9961 |             -0.9608 |                -0.975 |
| Precision, malicious = positive (%)    | n/a      |         99.992  |             96.645  |            n/a      |               n/a     |
| Recall / detection rate, malicious (%) | n/a      |         97.1363 |            100      |            n/a      |               n/a     |
| F1, malicious (%)                      | n/a      |         98.5434 |             98.2939 |            n/a      |               n/a     |
| Benign precision (%)                   | n/a      |         54.7412 |              0      |            n/a      |               n/a     |
| Benign recall = specificity (%)        | n/a      |         99.7763 |              0      |            n/a      |               n/a     |
| Benign F1 (%)                          | n/a      |         70.6959 |              0      |            n/a      |               n/a     |
| False positive rate (%)                |   9.17   |          0.2237 |            100      |             -8.9463 |               -97.561 |
| False negative rate (%)                | n/a      |          2.8637 |              0      |            n/a      |               n/a     |
| Matthews corr. coef.                   | n/a      |          0.7283 |              0      |            n/a      |               n/a     |
| ROC-AUC                                | n/a      |          0.995  |              0.5    |            n/a      |               n/a     |
| PR-AUC (average precision)             | n/a      |          0.9998 |              0.9664 |            n/a      |               n/a     |
| Cross-entropy loss                     |   0.0275 |          0.0744 |              0.5349 |              0.0469 |               170.545 |