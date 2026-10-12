"""
model.py — Model definitions for all 5 SOC estimation architectures.

Implemented models
------------------
1. Transformer  — Custom encoder with sinusoidal positional encoding
2. LSTM         — Regularised 2-layer LSTM
3. BiLSTM       — Regularised 2-layer Bidirectional LSTM
4. SVR          — Support Vector Regressor (sklearn)
5. Random Forest — Ensemble regressor (sklearn)

All deep-learning models take input shape (seq_length, n_features) and
return a single scalar SOC value (scaled, 0–1).  Inverse-transform with
scaler_y to get SOC in percent (0–100).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, Lambda, LSTM, Bidirectional,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from sklearn.svm import SVR as _SVR
from sklearn.ensemble import RandomForestRegressor

from src import config


# =============================================================================
# 1. Transformer
# =============================================================================

def _positional_encoding(seq_len: int, d_model: int) -> tf.Tensor:
    """Sinusoidal positional encoding (seq_len, d_model)."""
    positions = np.arange(seq_len)[:, np.newaxis]          # (T, 1)
    dims      = np.arange(d_model)[np.newaxis, :]           # (1, D)
    angles    = positions / np.power(10_000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)   # (1, T, D)


def _transformer_encoder_block(x, d_model, num_heads, ffn_dim, dropout_rate, block_idx):
    """Single Transformer encoder block: Self-Attention + FFN with residuals."""
    # --- Multi-Head Self-Attention ---
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        name=f"mha_{block_idx}",
    )(x, x)
    attn_output = Dropout(dropout_rate, name=f"attn_drop_{block_idx}")(attn_output)
    x = LayerNormalization(epsilon=1e-6, name=f"ln1_{block_idx}")(x + attn_output)

    # --- Feed-Forward Network ---
    ffn = Dense(ffn_dim, activation="relu", name=f"ffn1_{block_idx}")(x)
    ffn = Dense(d_model,                    name=f"ffn2_{block_idx}")(ffn)
    ffn = Dropout(dropout_rate, name=f"ffn_drop_{block_idx}")(ffn)
    x   = LayerNormalization(epsilon=1e-6, name=f"ln2_{block_idx}")(x + ffn)
    return x


def build_transformer_model(
    seq_len:      int   = config.SEQ_LENGTH,
    n_features:   int   = len(config.FEATURE_COLS),
    d_model:      int   = config.TRANSFORMER_D_MODEL,
    num_heads:    int   = config.TRANSFORMER_NUM_HEADS,
    ffn_dim:      int   = config.TRANSFORMER_FFN_DIM,
    num_layers:   int   = config.TRANSFORMER_NUM_LAYERS,
    dropout_rate: float = config.TRANSFORMER_DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> Model:
    """
    Lightweight Transformer Encoder for SOC regression.

    Architecture: Input projection → Positional encoding →
                  N × Encoder blocks → Last-token pooling →
                  Dense(64, ReLU) → Dense(1)
    """
    inputs = Input(shape=(seq_len, n_features), name="input_sequence")

    # Project raw features to d_model dimensions
    x = Dense(d_model, name="input_projection")(inputs)

    # Add sinusoidal positional encoding (broadcast over batch)
    x = x + _positional_encoding(seq_len, d_model)
    x = Dropout(dropout_rate, name="positional_dropout")(x)

    for i in range(num_layers):
        x = _transformer_encoder_block(x, d_model, num_heads, ffn_dim, dropout_rate, i)

    # Take representation at the last time-step
    x = Lambda(lambda t: t[:, -1, :], name="last_token")(x)
    x = Dense(64, activation="relu", name="regression_dense")(x)
    x = Dropout(dropout_rate, name="regression_dropout")(x)
    outputs = Dense(1, name="soc_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="SoC_Transformer_Encoder")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =============================================================================
# 2. LSTM
# =============================================================================

def build_lstm_model(
    seq_len:       int   = config.SEQ_LENGTH,
    n_features:    int   = len(config.FEATURE_COLS),
    units_1:       int   = config.LSTM_UNITS_1,
    units_2:       int   = config.LSTM_UNITS_2,
    dropout_rate:  float = config.LSTM_DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> Sequential:
    """
    Regularised 2-layer LSTM for SOC regression.

    L2 kernel + recurrent regularization reduces battery-specific overfitting.
    """
    l2  = tf.keras.regularizers.l2(1e-4)

    model = Sequential(name="LSTM_SoC_Regularized")
    model.add(Input(shape=(seq_len, n_features)))
    model.add(LSTM(
        units_1,
        return_sequences=True,
        kernel_regularizer=l2,
        recurrent_regularizer=l2,
    ))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(
        units_2,
        kernel_regularizer=l2,
        recurrent_regularizer=l2,
    ))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dropout(0.15))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =============================================================================
# 3. Bidirectional LSTM
# =============================================================================

def build_bilstm_model(
    seq_len:       int   = config.SEQ_LENGTH,
    n_features:    int   = len(config.FEATURE_COLS),
    units_1:       int   = config.BILSTM_UNITS_1,
    units_2:       int   = config.BILSTM_UNITS_2,
    dropout_rate:  float = config.BILSTM_DROPOUT,
    learning_rate: float = config.LEARNING_RATE,
) -> Sequential:
    """
    Regularised 2-layer Bidirectional LSTM for SOC regression.
    """
    l2  = tf.keras.regularizers.l2(1e-4)

    model = Sequential(name="BiLSTM_SoC_Regularized")
    model.add(Input(shape=(seq_len, n_features)))
    model.add(Bidirectional(LSTM(
        units_1,
        return_sequences=True,
        kernel_regularizer=l2,
        recurrent_regularizer=l2,
    )))
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(
        units_2,
        kernel_regularizer=l2,
        recurrent_regularizer=l2,
    )))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
    model.add(Dropout(0.15))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


# =============================================================================
# 4. SVR
# =============================================================================

def build_svr_model(
    C:       float = 10.0,
    epsilon: float = 0.001,
    gamma:   str   = "scale",
    kernel:  str   = "rbf",
) -> _SVR:
    """
    Support Vector Regressor for SOC estimation.

    Note: SVR operates on flattened sequences. Reshape (N, T, F) → (N, T*F)
    before calling fit/predict.
    """
    return _SVR(C=C, epsilon=epsilon, gamma=gamma, kernel=kernel)


# =============================================================================
# 5. Random Forest
# =============================================================================

def build_random_forest_model(
    n_estimators:     int = config.RF_N_ESTIMATORS,
    max_depth:        int = config.RF_MAX_DEPTH,
    min_samples_leaf: int = config.RF_MIN_SAMPLES_LEAF,
    seed:             int = config.SEED,
) -> RandomForestRegressor:
    """
    Random Forest Regressor for SOC estimation.

    Note: Operates on flattened sequences. Reshape (N, T, F) → (N, T*F)
    before calling fit/predict.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )


# =============================================================================
# Quick smoke-test
# =============================================================================

if __name__ == "__main__":
    import numpy as np

    seq_len    = config.SEQ_LENGTH
    n_features = len(config.FEATURE_COLS)
    batch_size = 4

    dummy_X  = np.random.rand(batch_size, seq_len, n_features).astype(np.float32)
    dummy_X_flat = dummy_X.reshape(batch_size, -1)

    print("=== Transformer ===")
    t = build_transformer_model()
    t.summary()
    print("Output shape:", t.predict(dummy_X, verbose=0).shape)

    print("\n=== LSTM ===")
    lstm = build_lstm_model()
    lstm.summary()
    print("Output shape:", lstm.predict(dummy_X, verbose=0).shape)

    print("\n=== BiLSTM ===")
    bilstm = build_bilstm_model()
    bilstm.summary()
    print("Output shape:", bilstm.predict(dummy_X, verbose=0).shape)

    print("\n=== SVR (sklearn) ===")
    svr = build_svr_model()
    svr.fit(dummy_X_flat, np.random.rand(batch_size))
    print("SVR output:", svr.predict(dummy_X_flat[:2]))

    print("\n=== Random Forest (sklearn) ===")
    rf = build_random_forest_model(n_estimators=10)
    rf.fit(dummy_X_flat, np.random.rand(batch_size))
    print("RF output:", rf.predict(dummy_X_flat[:2]))

    print("\nAll model smoke-tests passed.")