<div align="center">

<img src="assets/banner.png" alt="Battery SOC Estimation Banner" width="100%"/>

# Battery State of Charge (SOC) Estimation

**Master's Thesis · Ayoub HAMMAD · 2025–2026**

A comparative machine-learning framework for **SOC estimation** in Electric Vehicle (EV) Intelligent Battery Management Systems (iBMS), benchmarking five architectures on the NASA Battery Dataset under a strict zero data-leakage protocol.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20Battery-0B3D91?logo=nasa&logoColor=white)](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Key Highlights](#-key-highlights)
- [Results](#-results)
- [Visualizations](#️-visualizations)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Architecture Details](#️-architecture-details)
- [Experimental Protocol](#-experimental-protocol)
- [Data](#-data)
- [References](#-references)

---

## ✨ Key Highlights

- 🏆 **5-model benchmark** — Transformer, BiLSTM, LSTM, Random Forest, SVR
- ⚡ **Coulomb Counting ground truth** — dynamic SOC computed per discharge cycle (0–100 %)
- 🔒 **Zero data-leakage evaluation** — battery-level splits (11 train / 1 val / 1 test)
- 🪶 **Lightweight edge-ready Transformer** — optimized for embedded BMS microcontrollers
- 📊 **Full reproducibility** — all metrics, plots, and training code included

---

## 📊 Results

### Test Battery: B0006 (Held-Out)

| Model | RMSE (%) ↓ | MAE (%) ↓ | R² ↑ | Max Error (%) ↓ |
|:---|:---:|:---:|:---:|:---:|
| 🥇 **Transformer** | **5.33** | **4.16** | **0.9648** | 15.92 |
| 🥈 Random Forest | 4.33 | 3.09 | 0.9768 | 17.01 |
| 🥉 SVR | 9.21 | 7.07 | 0.8950 | 36.19 |
| BiLSTM | 9.50 | 6.86 | 0.8882 | 25.70 |
| LSTM | 10.61 | 7.63 | 0.8604 | 27.94 |

> SOC expressed in % (0–100). Lower RMSE/MAE and higher R² is better.

### Inference Time (100-run average, single sample)

| Model | Avg Time (ms) | Edge Suitable |
|:---|:---:|:---:|
| SVR | **6.25** | ✅ |
| Random Forest | 45.56 | ✅ |
| **Transformer** | 75.92 | ✅ |
| LSTM | 76.82 | ✅ |
| BiLSTM | 79.52 | ✅ |

---

## 🖼️ Visualizations

### All Models — SOC Prediction on Test Battery B0006

<div align="center">
<img src="assets/plots/soc_model_prediction_comparison_all_models.png" width="90%" alt="All models prediction comparison"/>
</div>

---

### Metrics Comparison

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_model_rmse_comparison.png" width="100%" alt="RMSE Comparison"/><br/><b>RMSE</b></td>
<td align="center"><img src="assets/plots/soc_model_mae_comparison.png" width="100%" alt="MAE Comparison"/><br/><b>MAE</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_model_r2_comparison.png" width="100%" alt="R² Comparison"/><br/><b>R²</b></td>
<td align="center"><img src="assets/plots/soc_model_max_error_comparison.png" width="100%" alt="Max Error Comparison"/><br/><b>Max Error</b></td>
</tr>
</table>
</div>

---

### Error Distribution Across All Models

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/all_models_error_boxplot.png" width="100%" alt="Error Boxplot"/><br/><b>Absolute Error Distribution (Box Plot)</b></td>
<td align="center"><img src="assets/plots/all_models_error_kde.png" width="100%" alt="Error KDE"/><br/><b>Error Density (KDE)</b></td>
</tr>
</table>
</div>

---

### Transformer

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_pred_transformer.png" width="100%" alt="Transformer Prediction"/><br/><b>SOC Prediction</b></td>
<td align="center"><img src="assets/plots/soc_transformer_predicted_vs_true.png" width="100%" alt="Transformer Predicted vs True"/><br/><b>Predicted vs True</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_transformer_loss_curve.png" width="100%" alt="Transformer Loss"/><br/><b>Training Loss Curve</b></td>
<td align="center"><img src="assets/plots/soc_transformer_absolute_error.png" width="100%" alt="Transformer Error"/><br/><b>Absolute Error</b></td>
</tr>
</table>
</div>

---

### LSTM & BiLSTM

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_lstm_prediction.png" width="100%" alt="LSTM Prediction"/><br/><b>LSTM — SOC Prediction</b></td>
<td align="center"><img src="assets/plots/soc_bilstm_prediction.png" width="100%" alt="BiLSTM Prediction"/><br/><b>BiLSTM — SOC Prediction</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_lstm_loss_curve.png" width="100%" alt="LSTM Loss"/><br/><b>LSTM — Training Loss</b></td>
<td align="center"><img src="assets/plots/soc_bilstm_loss_curve.png" width="100%" alt="BiLSTM Loss"/><br/><b>BiLSTM — Training Loss</b></td>
</tr>
</table>
</div>

---

### Random Forest & SVR

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_random_forest_prediction.png" width="100%" alt="RF Prediction"/><br/><b>Random Forest — SOC Prediction</b></td>
<td align="center"><img src="assets/plots/soc_svr_prediction.png" width="100%" alt="SVR Prediction"/><br/><b>SVR — SOC Prediction</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_random_forest_feature_importance.png" width="100%" alt="RF Feature Importance"/><br/><b>Feature Importance</b></td>
<td align="center"><img src="assets/plots/soc_svr_error_distribution.png" width="100%" alt="SVR Error"/><br/><b>SVR Error Distribution</b></td>
</tr>
</table>
</div>

---

### Coulomb Counting Ground Truth

<div align="center">
<img src="assets/plots/soc_coulomb_counted_sample.png" width="75%" alt="Coulomb Counting SOC"/>
<br/><i>Dynamic SOC computed via Coulomb Counting on a single discharge cycle</i>
</div>

---

## 📂 Repository Structure

```text
battery-soc-estimation/
├── assets/
│   ├── banner.png                       # Repository banner
│   ├── plots/                           # 39 result visualizations
│   └── thesis_ref/                      # Reference papers
├── results/                             # Benchmark CSVs (metrics, predictions, inference times)
│   ├── all_models_comparison_metrics.csv
│   ├── all_models_inference_time.csv
│   └── ...
├── models/
│   └── README.md                        # Model weight info & re-training guide
├── src/
│   ├── config.py          # Hyperparameters, paths, battery splits
│   ├── data_pipeline.py   # Data loading, Coulomb counting, sequence builder
│   ├── evaluate.py        # Shared evaluation utilities
│   ├── model.py           # All 5 model builders
│   └── train.py           # CLI training script
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone

```bash
git clone https://github.com/<your-username>/battery-soc-estimation.git
cd battery-soc-estimation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the NASA Battery Dataset

Get it from [Kaggle](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) and place it in `data/raw/`:

```
data/raw/
├── metadata.csv
└── data/
    ├── 05000.csv
    └── ...
```

### 4. Train

```bash
# All 5 models
python -m src.train

# Single model
python -m src.train --model transformer
python -m src.train --model lstm
python -m src.train --model bilstm
python -m src.train --model svr
python -m src.train --model random_forest
```

---

## 🏗️ Architecture Details

### Lightweight Transformer Encoder
```
Input (30 timesteps × 4 features)
  → Dense projection (d_model = 64)
  → Sinusoidal positional encoding
  → 2 × [Multi-Head Self-Attention (4 heads) + FFN (128 dim)] + Dropout (0.2)
  → Last-token readout
  → Dense(64, ReLU) → Dense(1)
```

### LSTM (Regularised)
```
Input → LSTM(48, L2) → Dropout(0.3) → LSTM(24, L2) → Dropout(0.3) → Dense(16, ReLU) → Dense(1)
```

### BiLSTM (Regularised)
```
Input → BiLSTM(40, L2) → Dropout(0.35) → BiLSTM(20, L2) → Dropout(0.35) → Dense(16, ReLU) → Dense(1)
```

### SVR & Random Forest
Both operate on **flattened sequences** (30 timesteps × 4 features = 120 features).

---

## 📐 Experimental Protocol

| Setting | Value |
|:---|:---|
| Dataset | NASA Battery Dataset (Li-Ion, 18650 cells) |
| Ground-truth SOC | Coulomb Counting (0–100 %) |
| Input features | Voltage, Current, Temperature, Δt |
| Sequence length | 30 timesteps |
| Train batteries | B0005, B0007, B0028, B0032, B0034, B0045–B0049, B0054 (11 cells) |
| Validation battery | B0018 |
| Test battery | B0006 |
| Scaler | MinMaxScaler — fit on train only |
| Optimizer | Adam (lr = 3×10⁻⁴) |
| Loss function | MSE |
| Max epochs | 100 (EarlyStopping patience = 12) |
| Batch size | 64 |
| Random seed | 42 |

---

## 📁 Data

The raw NASA Battery Dataset is **not included** (gitignored). Download from:

- 🔗 **Kaggle**: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset
- 🔗 **NASA PCoE**: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

---

## 📚 References

1. Saha, B. and Goebel, K. (2007). *Battery Data Set*. NASA Ames Prognostics Data Repository.
2. C. Li et al. (2014). *A Transformer-Based Model for State of Charge Estimation*. [`assets/thesis_ref/li2014.pdf`](assets/thesis_ref/li2014.pdf)
3. B. Xia et al. (2018). *State of charge estimation using optimized Levenberg-Marquardt wavelet neural network*. Energy.

---

<div align="center">

**Ayoub HAMMAD** — Master's Thesis 2025–2026

*If you find this useful, please consider giving it a ⭐*

</div>