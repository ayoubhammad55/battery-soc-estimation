"""
train.py — Standalone training script for all SOC estimation models.

Usage
-----
    python -m src.train                 # train all 5 models
    python -m src.train --model lstm    # train one model only
    python -m src.train --help

Output
------
Trained weights are saved to models/:
    best_soc_transformer.keras
    best_soc_lstm.keras
    best_soc_bilstm.keras
    best_soc_svr.joblib
    best_soc_random_forest.joblib
    scaler_X.joblib
    scaler_y.joblib
"""

import argparse
import random
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src import config
from src.data_pipeline import run_pipeline
from src.model import (
    build_bilstm_model,
    build_lstm_model,
    build_random_forest_model,
    build_svr_model,
    build_transformer_model,
)
from src.evaluate import evaluate_predictions, print_metrics_table


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seeds(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Shared Keras callbacks
# ---------------------------------------------------------------------------

def _keras_callbacks(checkpoint_path: str):
    return [
        EarlyStopping(
            monitor="val_loss", patience=12,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=5, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            checkpoint_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
    ]


# ---------------------------------------------------------------------------
# Individual trainers
# ---------------------------------------------------------------------------

def train_transformer(data: dict):
    print("\n" + "=" * 60)
    print("Training: Transformer")
    print("=" * 60)
    model     = build_transformer_model()
    ckpt_path = str(config.MODEL_DIR / "best_soc_transformer.keras")

    model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=_keras_callbacks(ckpt_path),
        verbose=1,
    )
    model = tf.keras.models.load_model(ckpt_path, safe_mode=False)
    print(f"Saved → {ckpt_path}")
    return model


def train_lstm(data: dict):
    print("\n" + "=" * 60)
    print("Training: LSTM")
    print("=" * 60)
    model     = build_lstm_model()
    ckpt_path = str(config.MODEL_DIR / "best_soc_lstm.keras")

    model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=_keras_callbacks(ckpt_path),
        verbose=1,
    )
    model = tf.keras.models.load_model(ckpt_path)
    print(f"Saved → {ckpt_path}")
    return model


def train_bilstm(data: dict):
    print("\n" + "=" * 60)
    print("Training: BiLSTM")
    print("=" * 60)
    model     = build_bilstm_model()
    ckpt_path = str(config.MODEL_DIR / "best_soc_bilstm.keras")

    model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=_keras_callbacks(ckpt_path),
        verbose=1,
    )
    model = tf.keras.models.load_model(ckpt_path)
    print(f"Saved → {ckpt_path}")
    return model


def train_svr(data: dict):
    print("\n" + "=" * 60)
    print("Training: SVR")
    print("=" * 60)
    rng = np.random.default_rng(config.SEED)

    X_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
    n      = min(config.SVR_MAX_TRAIN, len(X_flat))
    idx    = rng.choice(len(X_flat), size=n, replace=False)

    svr = build_svr_model()
    print(f"  Fitting SVR on {n:,} samples ...")
    t0 = time.perf_counter()
    svr.fit(X_flat[idx], data["y_train"][idx])
    print(f"  SVR training time: {time.perf_counter() - t0:.1f}s")

    out_path = str(config.MODEL_DIR / "best_soc_svr.joblib")
    joblib.dump(svr, out_path)
    print(f"Saved → {out_path}")
    return svr


def train_random_forest(data: dict):
    print("\n" + "=" * 60)
    print("Training: Random Forest")
    print("=" * 60)

    X_flat = data["X_train"].reshape(data["X_train"].shape[0], -1)
    rf     = build_random_forest_model()
    t0     = time.perf_counter()
    rf.fit(X_flat, data["y_train"])
    print(f"  RF training time: {time.perf_counter() - t0:.1f}s")

    out_path = str(config.MODEL_DIR / "best_soc_random_forest.joblib")
    joblib.dump(rf, out_path)
    print(f"Saved → {out_path}")
    return rf


# ---------------------------------------------------------------------------
# Save / load scalers
# ---------------------------------------------------------------------------

def save_scalers(scaler_X, scaler_y):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, config.MODEL_DIR / "scaler_X.joblib")
    joblib.dump(scaler_y, config.MODEL_DIR / "scaler_y.joblib")
    print(f"Scalers saved to {config.MODEL_DIR}/")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "transformer":    train_transformer,
    "lstm":           train_lstm,
    "bilstm":         train_bilstm,
    "svr":            train_svr,
    "random_forest":  train_random_forest,
}


def main():
    parser = argparse.ArgumentParser(description="Train SOC estimation models")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    args = parser.parse_args()

    _set_seeds()
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[1/3] Running data pipeline ...")
    data = run_pipeline()
    save_scalers(data["scaler_X"], data["scaler_y"])

    # ── Training ─────────────────────────────────────────────────────────────
    selected = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    print(f"\n[2/3] Training: {selected}")

    trained_models = {}
    for name in selected:
        trained_models[name] = MODEL_REGISTRY[name](data)

    # ── Evaluation ───────────────────────────────────────────────────────────
    print("\n[3/3] Evaluating all trained models ...")
    all_metrics = []
    scaler_y    = data["scaler_y"]

    for name, model in trained_models.items():
        is_sklearn = name in ("svr", "random_forest")
        X_val_in   = data["X_val"].reshape(data["X_val"].shape[0], -1)  if is_sklearn else data["X_val"]
        X_test_in  = data["X_test"].reshape(data["X_test"].shape[0], -1) if is_sklearn else data["X_test"]

        val_pred  = model.predict(X_val_in).ravel()  if is_sklearn else model.predict(X_val_in,  verbose=0).ravel()
        test_pred = model.predict(X_test_in).ravel() if is_sklearn else model.predict(X_test_in, verbose=0).ravel()

        val_m  = evaluate_predictions(f"{name} (val)",  data["y_val"],  val_pred,  scaler_y)
        test_m = evaluate_predictions(f"{name} (test)", data["y_test"], test_pred, scaler_y)
        all_metrics.extend([val_m, test_m])

    metrics_df = pd.DataFrame([
        {k: v for k, v in m.items() if k not in ("y_true", "y_pred")}
        for m in all_metrics
    ]).set_index("Model")

    print_metrics_table(metrics_df)

    out_path = config.RESULTS_DIR / "all_models_comparison_metrics.csv"
    metrics_df.round(4).to_csv(out_path)
    print(f"\nMetrics saved → {out_path}")


if __name__ == "__main__":
    main()
