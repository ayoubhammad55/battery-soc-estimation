"""
config.py — Central configuration for the battery-soc-estimation project.

All hyperparameters, directory paths, battery splits, and physical constants
are defined here so that every other module imports from a single source.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
BASE_DIR           = Path(__file__).resolve().parent.parent
DATA_RAW_DIR       = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR          = BASE_DIR / "models"
ASSETS_DIR         = BASE_DIR / "assets"
RESULTS_DIR        = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# Battery entity splits (zero data-leakage: split by battery, not by time)
# Train: 11 batteries | Val: B0018 | Test: B0006
# ---------------------------------------------------------------------------
TRAIN_BATTERIES = [
    "B0005", "B0007", "B0028", "B0032", "B0034",
    "B0045", "B0046", "B0047", "B0048", "B0049", "B0054",
]
VAL_BATTERIES  = ["B0018"]
TEST_BATTERIES = ["B0006"]

# ---------------------------------------------------------------------------
# Feature / target configuration
# ---------------------------------------------------------------------------
FEATURE_COLS = ["voltage", "current", "temperature", "dt"]
TARGET_COL   = "SOC"          # SOC is expressed in % (0–100)
SEQ_LENGTH   = 30             # Sliding-window sequence length (timesteps)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
NOMINAL_CAPACITY = 2.0        # Nominal cell capacity in Ampere-hours (Ah)

# ---------------------------------------------------------------------------
# Model hyperparameters (defaults — override in train.py as needed)
# ---------------------------------------------------------------------------
SEED          = 42
BATCH_SIZE    = 64
EPOCHS        = 100
LEARNING_RATE = 3e-4

# Transformer
TRANSFORMER_D_MODEL    = 64
TRANSFORMER_NUM_HEADS  = 4
TRANSFORMER_FFN_DIM    = 128
TRANSFORMER_NUM_LAYERS = 2
TRANSFORMER_DROPOUT    = 0.2

# LSTM / BiLSTM
LSTM_UNITS_1   = 48
LSTM_UNITS_2   = 24
LSTM_DROPOUT   = 0.3
BILSTM_UNITS_1 = 40
BILSTM_UNITS_2 = 20
BILSTM_DROPOUT = 0.35

# SVR
SVR_MAX_TRAIN = 80_000

# Random Forest
RF_N_ESTIMATORS   = 200
RF_MAX_DEPTH      = 20
RF_MIN_SAMPLES_LEAF = 3