 <div align="center">

<img src="assets/banner.png" alt="Battery SOC Estimation Banner" width="100%"/>

# Battery State of Charge (SOC) Estimation

**Ayoub HAMMAD · 2025–2026**

A machine-learning project developed as part of my master's thesis, focusing on **State of Charge (SOC) estimation** for Battery Management Systems (BMS). This repository benchmarks five machine-learning architectures on the NASA Battery Dataset, comparing prediction accuracy, generalization to a held-out battery, and inference time.

*This repository covers the SOC estimation component only. It does not contain the complete autonomous robot, intelligent BMS, or wireless charging system developed as part of the master's thesis.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow\&logoColor=white)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20Battery-0B3D91?logo=nasa\&logoColor=white)](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

</div>

---

## 📋 Table of Contents

* [Project Scope](#-project-scope)
* [Key Highlights](#-key-highlights)
* [Results](#-results)
* [Visualizations](#️-visualizations)
* [Repository Structure](#-repository-structure)
* [Quick Start](#-quick-start)
* [Architecture Details](#️-architecture-details)
* [Experimental Protocol](#-experimental-protocol)
* [Data](#-data)
* [References](#-references)

---

## 🎯 Project Scope

My master's thesis, *Design and Implementation of an Autonomous Service Robot with Intelligent BMS and Wireless Charging*, covered several interconnected engineering components.

This repository focuses specifically on the **machine-learning component for battery SOC estimation**.

The objective is to compare different model architectures for estimating battery SOC from electrical and thermal measurements. The project includes data processing, SOC reference calculation, model training, evaluation, and result visualization.

The broader thesis also covered the autonomous robot platform, battery monitoring, wireless charging, and system integration. Those components are outside the scope of this repository.

---

## ✨ Key Highlights

* **Five-model benchmark** — Transformer, BiLSTM, LSTM, Random Forest, and Support Vector Regression (SVR).
* **SOC reference estimation** — Coulomb Counting applied to battery discharge-cycle measurements.
* **Battery-level evaluation** — separate training, validation, and test batteries to assess generalization to an unseen battery.
* **Comprehensive evaluation** — RMSE, MAE, R², maximum absolute error, and inference time.
* **Deep learning and classical ML comparison** — sequence-based neural networks compared with traditional machine-learning models.
* **Reproducible workflow** — training scripts, evaluation code, result tables, and visualizations.

---

## 📊 Results

### Test Battery: B0006 (Held Out)

The following table presents the reported performance of the five models on test battery B0006.

| Model             | RMSE (%) ↓ | MAE (%) ↓ |       R² ↑ | Max Error (%) ↓ |
| :---------------- | ---------: | --------: | ---------: | --------------: |
| **Random Forest** |   **4.33** |  **3.09** | **0.9768** |           17.01 |
| Transformer       |       5.33 |      4.16 |     0.9648 |       **15.92** |
| SVR               |       9.21 |      7.07 |     0.8950 |           36.19 |
| BiLSTM            |       9.50 |      6.86 |     0.8882 |           25.70 |
| LSTM              |      10.61 |      7.63 |     0.8604 |           27.94 |

SOC errors are expressed in percentage points on a 0–100% SOC scale. Lower RMSE, MAE, and maximum absolute error indicate smaller prediction errors, while higher R² indicates a better fit to the reference values.

**Key observation:** Random Forest achieves the lowest RMSE and MAE and the highest R² in these results. The Transformer has the lowest maximum absolute error. The best model therefore depends on the evaluation metric and any deployment constraints.

### Inference Time

Reported average inference time over 100 runs for a single sample.

| Model         | Average Inference Time (ms) |
| :------------ | --------------------------: |
| SVR           |                        6.25 |
| Random Forest |                       45.56 |
| Transformer   |                       75.92 |
| LSTM          |                       76.82 |
| BiLSTM        |                       79.52 |

Inference time depends on the hardware, software environment, and measurement procedure. These measurements alone do not establish suitability for a particular embedded platform.

---

## 🖼️ Visualizations

### All Models — SOC Prediction on Test Battery B0006

<div align="center">
<img src="assets/plots/soc_model_prediction_comparison_all_models.png" width="90%" alt="SOC prediction comparison across all five models on test battery B0006"/>
</div>

---

### Metrics Comparison

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_model_rmse_comparison.png" width="100%" alt="RMSE comparison"/><br/><b>RMSE</b></td>
<td align="center"><img src="assets/plots/soc_model_mae_comparison.png" width="100%" alt="MAE comparison"/><br/><b>MAE</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_model_r2_comparison.png" width="100%" alt="R-squared comparison"/><br/><b>R²</b></td>
<td align="center"><img src="assets/plots/soc_model_max_error_comparison.png" width="100%" alt="Maximum error comparison"/><br/><b>Maximum Absolute Error</b></td>
</tr>
</table>
</div>

---

### Error Distribution Across Models

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/all_models_error_boxplot.png" width="100%" alt="Absolute error box plot"/><br/><b>Absolute Error Distribution</b></td>
<td align="center"><img src="assets/plots/all_models_error_kde.png" width="100%" alt="Error density distributions"/><br/><b>Error Density</b></td>
</tr>
</table>
</div>

---

### Transformer

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_pred_transformer.png" width="100%" alt="Transformer SOC prediction"/><br/><b>SOC Prediction</b></td>
<td align="center"><img src="assets/plots/soc_transformer_predicted_vs_true.png" width="100%" alt="Transformer predicted versus reference SOC"/><br/><b>Predicted vs. Reference SOC</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_transformer_loss_curve.png" width="100%" alt="Transformer training loss"/><br/><b>Training Loss</b></td>
<td align="center"><img src="assets/plots/soc_transformer_absolute_error.png" width="100%" alt="Transformer absolute error"/><br/><b>Absolute Error</b></td>
</tr>
</table>
</div>

---

### LSTM and BiLSTM

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_lstm_prediction.png" width="100%" alt="LSTM SOC prediction"/><br/><b>LSTM — SOC Prediction</b></td>
<td align="center"><img src="assets/plots/soc_bilstm_prediction.png" width="100%" alt="BiLSTM SOC prediction"/><br/><b>BiLSTM — SOC Prediction</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_lstm_loss_curve.png" width="100%" alt="LSTM training loss"/><br/><b>LSTM — Training Loss</b></td>
<td align="center"><img src="assets/plots/soc_bilstm_loss_curve.png" width="100%" alt="BiLSTM training loss"/><br/><b>BiLSTM — Training Loss</b></td>
</tr>
</table>
</div>

---

### Random Forest and SVR

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_random_forest_prediction.png" width="100%" alt="Random Forest SOC prediction"/><br/><b>Random Forest — SOC Prediction</b></td>
<td align="center"><img src="assets/plots/soc_svr_prediction.png" width="100%" alt="SVR SOC prediction"/><br/><b>SVR — SOC Prediction</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_random_forest_feature_importance.png" width="100%" alt="Random Forest feature importance"/><br/><b>Feature Importance</b></td>
<td align="center"><img src="assets/plots/soc_svr_error_distribution.png" width="100%" alt="SVR error distribution"/><br/><b>SVR Error Distribution</b></td>
</tr>
</table>
</div>

---

### Coulomb Counting SOC Reference

<div align="center">
<img src="assets/plots/soc_coulomb_counted_sample.png" width="75%" alt="Example SOC reference estimated using Coulomb Counting"/>
<br/><i>Example of SOC estimated using Coulomb Counting during a battery discharge cycle.</i>
</div>

Coulomb Counting estimates SOC by integrating current over time. Its accuracy depends on factors such as the initial SOC, current measurement accuracy, and capacity assumptions. In this project, it provides the reference SOC values used to train and evaluate the models; it should not be interpreted as an independently measured, error-free ground truth.

---

## 📂 Repository Structure

```text
battery-soc-estimation/
├── assets/
│   ├── banner.png
│   └── plots/
│       └── ...                       # Result visualizations
├── models/
│   └── README.md                     # Model information and retraining notes
├── notebooks/
│   └── soc_estimation_all_models.ipynb
├── results/
│   ├── all_models_comparison_metrics.csv
│   ├── all_models_inference_time.csv
│   ├── all_models_test_predictions.csv
│   └── ...                           # Additional evaluation results
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration and dataset paths
│   ├── data_pipeline.py              # Data loading and preprocessing
│   ├── evaluate.py                   # Evaluation utilities
│   ├── model.py                      # Model definitions
│   └── train.py                      # Training entry point
├── .gitignore
├── README.md
└── requirements.txt
```

The raw dataset and trained model files are not included in the repository. Dataset files must be downloaded separately.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ayoubhammad55/battery-soc-estimation.git
cd battery-soc-estimation
```

### 2. Install dependencies

Python 3.10 or later is recommended, subject to the package versions specified in `requirements.txt`.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the NASA Battery Dataset

Download the dataset from [Kaggle — NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset).

Extract the files into the directory expected by `src/config.py`. The intended layout is:

```text
data/
└── raw/
    ├── metadata.csv
    └── data/
        ├── 05000.csv
        └── ...
```

Confirm the exact filenames, directory structure, and configured data path against the downloaded dataset and `src/config.py` before training.

### 4. Train the models

Train all configured models:

```bash
python -m src.train
```

Train a specific model:

```bash
python -m src.train --model transformer
python -m src.train --model lstm
python -m src.train --model bilstm
python -m src.train --model svr
python -m src.train --model random_forest
```

These commands assume that the training script supports the specified options and model names.

---

## 🏗️ Architecture Details

### Transformer Encoder

The reported Transformer configuration uses sequences of 30 timesteps, with four input features per timestep.

```text
Input (30 timesteps × 4 features)
  → Dense projection (d_model = 64)
  → Sinusoidal positional encoding
  → 2 × [Multi-Head Self-Attention (4 heads)
         + Feed-Forward Network (128 dimensions)]
  → Dropout (0.2)
  → Last-token readout
  → Dense(64, ReLU)
  → Dense(1)
```

### LSTM

```text
Input
  → LSTM(48, L2 regularization)
  → Dropout(0.3)
  → LSTM(24, L2 regularization)
  → Dropout(0.3)
  → Dense(16, ReLU)
  → Dense(1)
```

### BiLSTM

```text
Input
  → BiLSTM(40, L2 regularization)
  → Dropout(0.35)
  → BiLSTM(20, L2 regularization)
  → Dropout(0.35)
  → Dense(16, ReLU)
  → Dense(1)
```

### Support Vector Regression and Random Forest

SVR and Random Forest operate on flattened input sequences. With 30 timesteps and four features per timestep, each input sample contains 120 features.

---

## 📐 Experimental Protocol

The following settings summarize the reported experimental configuration. Verify them against `src/config.py` and the training code before treating them as a definitive reproduction specification.

| Setting            | Value                                                 |
| :----------------- | :---------------------------------------------------- |
| Dataset            | NASA Battery Dataset                                  |
| SOC reference      | Coulomb Counting                                      |
| Input features     | Voltage, current, temperature, and elapsed time (Δt)  |
| Sequence length    | 30 timesteps                                          |
| Training batteries | B0005, B0007, B0028, B0032, B0034, B0045–B0049, B0054 |
| Validation battery | B0018                                                 |
| Test battery       | B0006                                                 |
| Feature scaling    | MinMaxScaler, fitted on training data only            |
| Optimizer          | Adam                                                  |
| Learning rate      | 3 × 10⁻⁴                                              |
| Loss function      | Mean Squared Error (MSE)                              |
| Maximum epochs     | 100                                                   |
| Early stopping     | Patience of 12 epochs                                 |
| Batch size         | 64                                                    |
| Random seed        | 42                                                    |

### Evaluation and Data Leakage

Separating training, validation, and test data by battery helps evaluate generalization to a battery not used for model fitting. The scaler should be fitted on training data only, and test-battery information should not influence preprocessing, training, or model selection.

A complete leakage assessment also requires checking sequence construction, preprocessing that learns from data, hyperparameter tuning, and overlapping time windows. The battery-level split is an important safeguard, but it should not be treated as proof of zero leakage without verifying the implementation.

---

## 📁 Data

The raw NASA Battery Dataset is **not included in this repository**.

Download it from:

* **Kaggle:** [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
* **NASA PCoE:** [Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

After downloading, place the data in the directory configured by the project. Check `src/config.py` if your local directory structure differs from the example above.

---

## 📚 References

1. Saha, B., and Goebel, K. (2007). *Battery Data Set*. NASA Ames Prognostics Data Repository. [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

2. Xia, B., et al. (2018). *State of charge estimation using optimized Levenberg–Marquardt wavelet neural network*. **Energy**. Verify the complete bibliographic details against the original publication before using this citation in formal research.

---

<div align="center">

**Ayoub HAMMAD** · Master's Thesis Project Component · 2025–2026

*Feedback and suggestions are welcome.*

</div>
