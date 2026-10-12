"""
data_pipeline.py — End-to-end data loading, SOC computation, scaling, and
sequence generation for the battery-soc-estimation project.

Pipeline steps
--------------
1. Discover metadata.csv and numbered cycle CSVs from DATA_RAW_DIR.
2. Filter for discharge cycles of the selected batteries.
3. Compute ground-truth SOC via Coulomb Counting (0–100 %).
4. Add dt (time-step) feature.
5. Split by battery entity (train / val / test) — zero data leakage.
6. Fit MinMaxScaler on train only, then transform val & test.
7. Build sliding-window sequences of length SEQ_LENGTH.
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import config


# ---------------------------------------------------------------------------
# 1. Data discovery & loading
# ---------------------------------------------------------------------------

def _discover_files(raw_dir: Path):
    """Return (metadata_path, csv_map) where csv_map maps filename → Path."""
    meta_candidates = list(raw_dir.glob("**/metadata.csv"))
    if not meta_candidates:
        raise FileNotFoundError(
            f"No metadata.csv found under {raw_dir}. "
            "Download the NASA battery dataset and place it in data/raw/."
        )
    metadata_path = meta_candidates[0]
    dataset_dir   = metadata_path.parent

    all_csvs = glob.glob(str(dataset_dir / "**" / "*.csv"), recursive=True)
    csv_map  = {Path(p).name: Path(p) for p in all_csvs}
    return metadata_path, csv_map


def load_raw_discharge(batteries: list[str] | None = None) -> pd.DataFrame:
    """
    Load raw discharge rows for the requested batteries.

    Parameters
    ----------
    batteries : list[str] or None
        Battery IDs to load (e.g. ['B0005', 'B0006']). Defaults to all
        batteries defined in config (train + val + test).

    Returns
    -------
    pd.DataFrame
        Columns: voltage, current, temperature, time, capacity,
                 battery_name, cycle
    """
    if batteries is None:
        batteries = (
            config.TRAIN_BATTERIES
            + config.VAL_BATTERIES
            + config.TEST_BATTERIES
        )

    metadata_path, csv_map = _discover_files(config.DATA_RAW_DIR)

    metadata = pd.read_csv(metadata_path)
    metadata.columns = [c.lower().strip() for c in metadata.columns]

    required = {"type", "battery_id", "filename", "capacity"}
    missing  = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {missing}")

    discharge = metadata[metadata["type"].str.lower() == "discharge"].copy()
    discharge["capacity"] = pd.to_numeric(discharge["capacity"], errors="coerce")
    discharge = discharge.dropna(subset=["capacity"])
    discharge = discharge[discharge["battery_id"].isin(batteries)].copy()

    sort_cols = ["battery_id"] + [c for c in ["uid", "test_id"] if c in discharge.columns]
    discharge = discharge.sort_values(sort_cols)
    discharge["cycle"] = discharge.groupby("battery_id").cumcount() + 1

    rows = []
    for _, meta_row in discharge.iterrows():
        fname     = Path(str(meta_row["filename"])).name
        file_path = csv_map.get(fname)
        if file_path is None or not file_path.exists():
            continue

        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.rename(columns={
            "voltage_measured":     "voltage",
            "current_measured":     "current",
            "temperature_measured": "temperature",
            "time":                 "time",
        })

        needed = ["voltage", "current", "temperature", "time"]
        if not all(c in df.columns for c in needed):
            continue

        df = df[needed].copy()
        for col in needed:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=needed)

        df["capacity"]     = float(meta_row["capacity"])
        df["battery_name"] = meta_row["battery_id"]
        df["cycle"]        = int(meta_row["cycle"])
        rows.append(df)

    if not rows:
        raise ValueError(
            "No discharge CSV files were loaded. "
            "Check that metadata filenames match the files in data/raw/."
        )

    raw_df = pd.concat(rows, ignore_index=True)
    raw_df = raw_df.sort_values(["battery_name", "cycle", "time"]).reset_index(drop=True)

    print(f"[load_raw_discharge] Rows loaded: {len(raw_df):,}")
    print(f"[load_raw_discharge] Batteries:   {sorted(raw_df['battery_name'].unique())}")
    return raw_df


# ---------------------------------------------------------------------------
# 2. Coulomb Counting — ground-truth SOC in 0–100 %
# ---------------------------------------------------------------------------

def _compute_soc_coulomb(group: pd.DataFrame) -> pd.DataFrame:
    """Apply Coulomb counting SOC to a single (battery, cycle) group."""
    group = group.sort_values("time").copy()
    dt    = group["time"].diff().fillna(0).clip(lower=0)          # seconds

    charge_consumed_as = (group["current"].abs() * dt).cumsum()   # A·s
    q_nominal_as       = group["capacity"].iloc[0] * 3600.0       # Ah → A·s

    if q_nominal_as <= 0:
        group["SOC"] = np.nan
        group["dt"]  = dt
        return group

    group["SOC"] = (1.0 - charge_consumed_as / q_nominal_as).clip(0.0, 1.0) * 100.0
    group["dt"]  = dt
    return group


def compute_soc(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ground-truth SOC (%) and dt (s) columns to raw_df in-place.

    Returns
    -------
    pd.DataFrame
        Extended DataFrame with 'SOC' and 'dt' columns.
    """
    df = raw_df.groupby(["battery_name", "cycle"], group_keys=False).apply(
        _compute_soc_coulomb
    )
    df = df.dropna(subset=["SOC"]).reset_index(drop=True)
    print(f"[compute_soc] Rows after SOC calc: {len(df):,}")
    print(f"[compute_soc] SOC range: {df['SOC'].min():.2f}% – {df['SOC'].max():.2f}%")
    return df


# ---------------------------------------------------------------------------
# 3. Battery-level train / val / test split (zero leakage)
# ---------------------------------------------------------------------------

def split_by_battery(df: pd.DataFrame):
    """
    Split df into (train_df, val_df, test_df) using battery-level splits
    defined in config. No time-series overlap is possible across splits.
    """
    train_df = df[df["battery_name"].isin(config.TRAIN_BATTERIES)].copy()
    val_df   = df[df["battery_name"].isin(config.VAL_BATTERIES)].copy()
    test_df  = df[df["battery_name"].isin(config.TEST_BATTERIES)].copy()

    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        assert len(part) > 0, f"{name} split is empty — check battery IDs in config."

    print(f"[split] Train rows: {len(train_df):,}  Val rows: {len(val_df):,}  Test rows: {len(test_df):,}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 4. Scaling — fit on train only to prevent leakage
# ---------------------------------------------------------------------------

def scale_features(train_df, val_df, test_df):
    """
    Fit MinMaxScaler on train, transform val and test.

    Returns
    -------
    (train_df, val_df, test_df, scaler_X, scaler_y)
        DataFrames have FEATURE_COLS scaled in-place; 'SOC_scaled' column added.
    """
    feature_cols = config.FEATURE_COLS
    target_col   = config.TARGET_COL

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_df = train_df.copy()
    val_df   = val_df.copy()
    test_df  = test_df.copy()

    train_df[feature_cols]  = scaler_X.fit_transform(train_df[feature_cols])
    train_df["SOC_scaled"]  = scaler_y.fit_transform(train_df[[target_col]])

    val_df[feature_cols]    = scaler_X.transform(val_df[feature_cols])
    val_df["SOC_scaled"]    = scaler_y.transform(val_df[[target_col]])

    test_df[feature_cols]   = scaler_X.transform(test_df[feature_cols])
    test_df["SOC_scaled"]   = scaler_y.transform(test_df[[target_col]])

    return train_df, val_df, test_df, scaler_X, scaler_y


# ---------------------------------------------------------------------------
# 5. Sliding-window sequence builder
# ---------------------------------------------------------------------------

def build_sequences(df: pd.DataFrame, seq_length: int | None = None):
    """
    Build sliding-window (X, y) arrays grouped by (battery_name, cycle).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain scaled FEATURE_COLS and 'SOC_scaled'.
    seq_length : int
        Window length. Defaults to config.SEQ_LENGTH.

    Returns
    -------
    (X, y, meta)
        X     — float32 array of shape (N, seq_length, n_features)
        y     — float32 array of shape (N,)
        meta  — DataFrame with columns [battery_name, cycle, time]
    """
    if seq_length is None:
        seq_length = config.SEQ_LENGTH

    feature_cols = config.FEATURE_COLS
    X_all, y_all, meta_all = [], [], []

    for (battery, cycle), group in df.groupby(["battery_name", "cycle"]):
        group      = group.sort_values("time")
        X_values   = group[feature_cols].to_numpy(dtype=np.float32)
        y_values   = group["SOC_scaled"].to_numpy(dtype=np.float32)
        time_values = group["time"].to_numpy()

        if len(group) <= seq_length:
            continue

        for i in range(len(group) - seq_length):
            target_idx = i + seq_length
            X_all.append(X_values[i:target_idx])
            y_all.append(y_values[target_idx])
            meta_all.append((battery, int(cycle), float(time_values[target_idx])))

    X   = np.array(X_all, dtype=np.float32)
    y   = np.array(y_all, dtype=np.float32)
    meta = pd.DataFrame(meta_all, columns=["battery_name", "cycle", "time"])
    return X, y, meta


# ---------------------------------------------------------------------------
# 6. High-level convenience function
# ---------------------------------------------------------------------------

def run_pipeline():
    """
    Run the full data pipeline from raw files to ready-to-train arrays.

    Returns
    -------
    dict with keys:
        X_train, y_train, train_meta,
        X_val,   y_val,   val_meta,
        X_test,  y_test,  test_meta,
        scaler_X, scaler_y
    """
    raw_df = load_raw_discharge()
    df     = compute_soc(raw_df)

    train_df, val_df, test_df = split_by_battery(df)
    train_df, val_df, test_df, scaler_X, scaler_y = scale_features(
        train_df, val_df, test_df
    )

    print("\n[build_sequences] Building sliding-window sequences ...")
    X_train, y_train, train_meta = build_sequences(train_df)
    X_val,   y_val,   val_meta   = build_sequences(val_df)
    X_test,  y_test,  test_meta  = build_sequences(test_df)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

    return dict(
        X_train=X_train, y_train=y_train, train_meta=train_meta,
        X_val=X_val,     y_val=y_val,     val_meta=val_meta,
        X_test=X_test,   y_test=y_test,   test_meta=test_meta,
        scaler_X=scaler_X, scaler_y=scaler_y,
    )


if __name__ == "__main__":
    data = run_pipeline()
    print("\nPipeline completed successfully.")