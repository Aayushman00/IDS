| variant                   |   size_mb |   ms_per_sample |   accuracy |   accuracy_drop_pp |
|:--------------------------|----------:|----------------:|-----------:|-------------------:|
| Keras float32 (GPU)       |    1.0428 |         15.2625 |     0.9875 |             0.0000 |
| TFLite float32            |    0.4195 |          0.0844 |     0.9875 |             0.0000 |
| TFLite dynamic-range int8 |    0.1967 |          0.0374 |     0.9876 |            -0.0050 |