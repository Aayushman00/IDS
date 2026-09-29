"""CNN-LSTM (paper Fig. 2) plus the baselines used to justify it.

All Keras models end in a K-way softmax (K=2 for binary), exactly like the
paper's 2-unit output layer, so one loss (sparse categorical cross-entropy)
serves every task and P(malicious) is simply ``proba[:, 1]``.
"""
from __future__ import annotations

import keras
from keras import layers
from sklearn.ensemble import RandomForestClassifier

from config import Config

EMBEDDING_LAYER = "lstm_embedding"   # used for the t-SNE plot


def _output(x, n_classes: int, name: str = "output"):
    # float32 output keeps softmax numerically safe under mixed precision
    return layers.Dense(n_classes, activation="softmax", dtype="float32", name=name)(x)


def _cnn_trunk(inp, cfg: Config):
    """Conv1D(64,k3)-BN-AvgPool-Conv1D(64,k3)-AvgPool-Flatten-Dense32-Dense16 (Fig. 2).

    'valid' padding + pool 2 reproduces the paper's Flatten size of 576
    (= 9 x 64) for 45 input features.
    """
    x = layers.Conv1D(cfg.conv_filters, cfg.conv_kernel, activation="relu", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.AveragePooling1D(cfg.pool_size, name="avgpool1")(x)
    x = layers.Conv1D(cfg.conv_filters, cfg.conv_kernel, activation="relu", name="conv2")(x)
    x = layers.AveragePooling1D(cfg.pool_size, name="avgpool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(cfg.dense_units[0], activation="relu", name="dense32")(x)
    if cfg.dropout:
        x = layers.Dropout(cfg.dropout)(x)
    return layers.Dense(cfg.dense_units[1], activation="relu", name="dense16")(x)


def build_paper_cnn_lstm(n_features: int, n_classes: int, cfg: Config) -> keras.Model:
    """Two-branch CNN-LSTM exactly as drawn in Fig. 2 of the paper.

    Branch A (classification task): Dense16 -> Dense(2, softmax).
    Branch B (prediction task):     Dense16 -> Reshape(16,1) -> LSTM(64) x2
                                    -> Dense(2, sigmoid).
    Both are concatenated (4 units) -> Dense(2, softmax) final output.
    """
    inp = keras.Input((n_features, 1), name="features")
    x = _cnn_trunk(inp, cfg)
    branch_a = layers.Dense(n_classes, activation="softmax", name="cnn_head")(x)
    r = layers.Reshape((cfg.dense_units[1], 1), name="reshape")(x)
    r = layers.LSTM(cfg.lstm_units, return_sequences=True, name="lstm1")(r)
    r = layers.LSTM(cfg.lstm_units, name=EMBEDDING_LAYER)(r)
    branch_b = layers.Dense(n_classes, activation="sigmoid", name="lstm_head")(r)
    merged = layers.Concatenate(name="concat")([branch_a, branch_b])
    return keras.Model(inp, _output(merged, n_classes), name="CNN-LSTM")


def build_generic_cnn_lstm(n_features: int, n_classes: int, cfg: Config) -> keras.Model:
    """Sequential CNN -> LSTM default from the project brief (--architecture generic)."""
    inp = keras.Input((n_features, 1), name="features")
    x = inp
    for filters in (64, 128):
        x = layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, name=EMBEDDING_LAYER)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, _output(x, n_classes), name="CNN-LSTM")


def build_cnn(n_features: int, n_classes: int, cfg: Config) -> keras.Model:
    """Ablation: the paper's CNN trunk without the LSTM branch."""
    inp = keras.Input((n_features, 1), name="features")
    return keras.Model(inp, _output(_cnn_trunk(inp, cfg), n_classes), name="CNN")


def build_lstm(n_features: int, n_classes: int, cfg: Config) -> keras.Model:
    """Ablation: the paper's two stacked LSTMs directly on the feature sequence."""
    inp = keras.Input((n_features, 1), name="features")
    x = layers.LSTM(cfg.lstm_units, return_sequences=True)(inp)
    x = layers.LSTM(cfg.lstm_units, name=EMBEDDING_LAYER)(x)
    return keras.Model(inp, _output(x, n_classes), name="LSTM")


def build_mlp(n_features: int, n_classes: int, cfg: Config) -> keras.Model:
    """Plain feed-forward reference with no convolution or recurrence."""
    inp = keras.Input((n_features, 1), name="features")
    x = layers.Flatten()(inp)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    return keras.Model(inp, _output(x, n_classes), name="MLP")


def build_main_model(n_features: int, n_classes: int, cfg: Config) -> keras.Model:
    builder = {"paper": build_paper_cnn_lstm, "generic": build_generic_cnn_lstm}[cfg.architecture]
    return builder(n_features, n_classes, cfg)


BASELINE_BUILDERS = {"CNN": build_cnn, "LSTM": build_lstm, "MLP": build_mlp}


def build_random_forest(cfg: Config, use_class_weight: bool) -> RandomForestClassifier:
    """Classical ML reference; same train split, no reshaping needed."""
    return RandomForestClassifier(
        n_estimators=cfg.rf_estimators, n_jobs=-1, random_state=cfg.seed,
        class_weight="balanced" if use_class_weight else None,
    )
