# Model Weights

This directory stores trained model weights and fitted scalers.

> **Note:** These files are excluded from version control (`.gitignore`) due to their size.
> To obtain them, either **re-train from scratch** using the training script,
> or download them from the shared link below (if provided by the author).

---

## Expected files after training

| File | Description | Approx. size |
|------|-------------|-------------|
| `best_soc_transformer.keras` | Trained Lightweight Transformer | ~1.5 MB |
| `best_soc_lstm.keras` | Trained LSTM | ~0.5 MB |
| `best_soc_bilstm.keras` | Trained BiLSTM | ~0.8 MB |
| `best_soc_svr.joblib` | Trained SVR | ~50–200 MB |
| `best_soc_random_forest.joblib` | Trained Random Forest | ~100–500 MB |
| `scaler_X.joblib` | MinMaxScaler for input features | < 1 KB |
| `scaler_y.joblib` | MinMaxScaler for SOC target | < 1 KB |

---

## How to re-train

```bash
# From the repository root:

# Train all models
python -m src.train

# Train a specific model
python -m src.train --model transformer
python -m src.train --model lstm
python -m src.train --model bilstm
python -m src.train --model svr
python -m src.train --model random_forest
```

Training requires the NASA Battery Dataset to be placed in `data/raw/`
(see the main [README](../README.md) for download instructions).

---

## Loading a saved model

```python
import tensorflow as tf
import joblib

# Deep learning models
transformer = tf.keras.models.load_model("models/best_soc_transformer.keras", safe_mode=False)
lstm        = tf.keras.models.load_model("models/best_soc_lstm.keras")
bilstm      = tf.keras.models.load_model("models/best_soc_bilstm.keras")

# sklearn models
svr    = joblib.load("models/best_soc_svr.joblib")
rf     = joblib.load("models/best_soc_random_forest.joblib")

# Scalers
scaler_X = joblib.load("models/scaler_X.joblib")
scaler_y = joblib.load("models/scaler_y.joblib")
```
