<div align="center">

<img src="assets/banner.png" alt="Battery SOC Estimation Banner" width="100%"/>

# Battery State of Charge (SOC) Estimation

**Ayoub HAMMAD · 2025–2026**

A machine-learning project developed as part of my master's thesis, focusing on **State of Charge (SOC) estimation** for Battery Management Systems (BMS). This repository benchmarks five machine-learning models on the NASA Battery Dataset, comparing prediction accuracy on a held-out battery and reported inference time.

*This repository covers the SOC estimation component only. It does not contain the complete autonomous robot, intelligent BMS, or wireless charging system developed as part of the master's thesis.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NASA Dataset](https://img.shields.io/badge/Dataset-NASA%20Battery-0B3D91?logo=nasa&logoColor=white)](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)

</div>

---

## 📋 Table of Contents

- [Project Scope](#-project-scope)
- [Key Highlights](#-key-highlights)
- [Results](#-results)
- [Visualizations](#️-visualizations)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Architecture Details](#️-architecture-details)
- [Experimental Protocol](#-experimental-protocol)
- [Data](#-data)
- [References](#-references)
- [License](#-license)

---

## 🎯 Project Scope

My master's thesis, *Design and Implementation of an Autonomous Service Robot with Intelligent BMS and Wireless Charging*, covered several interconnected engineering components.

This repository focuses specifically on the **machine-learning component for battery SOC estimation**.

The objective is to compare different model architectures for estimating battery SOC from electrical and thermal measurements. The workflow includes data processing, reference SOC calculation using Coulomb Counting, model training, evaluation, and result visualization.

The broader thesis also covered the autonomous robot platform, battery monitoring, wireless charging, and system integration. Those components are outside the scope of this repository.

---

## ✨ Key Highlights

- **Five-model benchmark:** Transformer, LSTM, BiLSTM, Random Forest, and Support Vector Regression (SVR).
- **SOC reference calculation:** Coulomb Counting applied to battery discharge-cycle measurements.
- **Battery-level evaluation:** Separate training, validation, and test batteries to evaluate performance on an unseen battery.
- **Multiple evaluation metrics:** RMSE, MAE, R², maximum absolute error, and reported inference time.
- **Deep learning versus classical ML:** Sequence-based neural networks compared with traditional machine-learning models.
- **Reproducible workflow:** Training scripts, evaluation utilities, result tables, and visualizations.
- **Deployment considerations:** Model performance and artifact size can be considered together when evaluating deployment options.

---

## 📊 Results

### Test Battery: B0006 (Held Out)

The following results are from the reported evaluation on test battery B0006.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Max Error ↓ |
|:--|--:|--:|--:|--:|
| **Random Forest** | **4.33** | **3.09** | **0.9768** | 17.01 |
| Transformer | 5.33 | 4.16 | 0.9648 | **15.92** |
| SVR | 9.21 | 7.07 | 0.8950 | 36.19 |
| BiLSTM | 9.50 | 6.86 | 0.8882 | 25.70 |
| LSTM | 10.61 | 7.63 | 0.8604 | 27.94 |

RMSE, MAE, and maximum error are expressed in **SOC percentage points** on a 0–100% scale. Lower error values are better, while a higher R² indicates a better fit to the reference values.

**Key observations**

- **Random Forest** achieves the lowest test RMSE and MAE and the highest R² among the five models.
- **Transformer** has the lowest maximum absolute error in this evaluation.
- **SVR** performs better on the test battery than either recurrent neural network, although its validation results differ substantially from its test results.
- The best model depends on the evaluation metric, model size, computational requirements, and deployment environment.

These results evaluate predictions against SOC reference values calculated using Coulomb Counting. They should not be interpreted as measurements against an independently established, error-free SOC ground truth.

### Validation results

Validation was performed on battery B0018.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Max Error ↓ |
|:--|--:|--:|--:|--:|
| Transformer | 3.18 | 2.15 | 0.9867 | 13.76 |
| LSTM | 3.78 | 2.98 | 0.9811 | 10.64 |
| BiLSTM | 5.20 | 3.96 | 0.9643 | 16.41 |
| Random Forest | 7.26 | 5.24 | 0.9303 | 23.41 |
| SVR | 31.54 | 26.65 | -0.3145 | 60.21 |

SVR's validation performance is substantially worse than its test performance. This suggests that performance may vary across batteries and warrants further investigation before drawing conclusions about generalization.

### Reported inference time

| Model | Average Inference Time (ms) |
|:--|--:|
| SVR | 6.25 |
| Random Forest | 45.56 |
| Transformer | 75.92 |
| LSTM | 76.82 |
| BiLSTM | 79.52 |

These are the reported average inference times for a single sample over 100 runs. The hardware and detailed timing methodology are not documented here, so the measurements should be treated as benchmark results rather than verified latency on a specific embedded platform.

Inference time can depend on the hardware, software environment, model implementation, preprocessing, and measurement procedure.

---

## 🖼️ Visualizations

### All models: SOC prediction on test battery B0006

<div align="center">
<img src="assets/plots/soc_model_prediction_comparison_all_models.png" width="90%" alt="SOC prediction comparison across all five models on test battery B0006"/>
</div>

---

### Metrics comparison

<div align="center">
<table>
<tr>
<td align="center"><img src="assets/plots/soc_model_rmse_comparison.png" width="100%" alt="RMSE comparison"/><br/><b>RMSE</b></td>
<td align="center"><img src="assets/plots/soc_model_mae_comparison.png" width="100%" alt="MAE comparison"/><br/><b>MAE</b></td>
</tr>
<tr>
<td align="center"><img src="assets/plots/soc_model_r2_comparison.png" width="100%" alt="R-squared comparison"/><br/><b>R²</b></td>
<td align="center"><img src="assets/plots/soc_model_max_error_comparison.png" width="100%" alt="Maximum absolute error comparison"/><br/><b>Maximum Absolute Error</b></td>
</tr>
</table>
</div>

---

### Error distributions

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

### Coulomb Counting SOC reference

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
│   └── README.md                     # Model artifacts and loading notes
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
├── LICENSE
├── README.md
└── requirements.txt
```

The raw dataset and trained model files are excluded from version control. Download the dataset separately before running the training pipeline. Trained model artifacts can be regenerated using the training script.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ayoubhammad55/battery-soc-estimation.git
cd battery-soc-estimation
```

### 2. Set up the environment

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

Place the extracted dataset in the location configured in `src/config.py`. The expected layout is:

```text
data/
└── raw/
    ├── metadata.csv
    └── data/
        ├── 05000.csv
        └── ...
```

The exact directory structure and filenames must match the dataset loader's expectations.

### 4. Train the models

Train all five models:

```bash
python -m src.train
```

Train a single model:

```bash
python -m src.train --model transformer
python -m src.train --model lstm
python -m src.train --model bilstm
python -m src.train --model svr
python -m src.train --model random_forest
```

To see the available options:

```bash
python -m src.train --help
```

The training script runs the data pipeline, saves the fitted scalers, trains the selected model or models, evaluates them on the validation and test sets, and writes the comparison metrics to `results/all_models_comparison_metrics.csv`.

**Note:** Training the full benchmark can require substantial memory and processing time, particularly for Random Forest and SVR. The original saved model files are not included in the repository.

---

## 🏗️ Architecture Details

### Transformer Encoder

The Transformer processes a sequence of 30 timesteps with four input features per timestep.

```text
Input (30 timesteps × 4 features)
 → Dense projection (d_model = 64)
 → Sinusoidal positional encoding
 → 2 × [
      Multi-Head Self-Attention (4 heads)
      Feed-Forward Network (128 dimensions)
   ]
 → Dropout (0.2)
 → Last-token readout
 → Dense(64, ReLU)
 → Dense(1)
```

### LSTM

```text
Input sequence
 → LSTM(48, L2 regularization)
 → Dropout(0.3)
 → LSTM(24, L2 regularization)
 → Dropout(0.3)
 → Dense(16, ReLU)
 → Dense(1)
```

### BiLSTM

```text
Input sequence
 → BiLSTM(40, L2 regularization)
 → Dropout(0.35)
 → BiLSTM(20, L2 regularization)
 → Dropout(0.35)
 → Dense(16, ReLU)
 → Dense(1)
```

### Support Vector Regression and Random Forest

SVR and Random Forest use flattened input sequences. Each sample therefore contains 120 features: 30 timesteps multiplied by four input features.

The SVR is trained on a deterministic random subset of at most 80,000 training windows. Random Forest uses the generated training windows.

---

## 📐 Experimental Protocol

The following table summarizes the configuration in the project code.

| Setting | Value |
|:--|:--|
| Dataset | NASA Battery Dataset |
| SOC reference | Coulomb Counting |
| Input features | Voltage, current, temperature, and elapsed time (Δt) |
| Sequence length | 30 timesteps |
| Training batteries | B0005, B0007, B0028, B0032, B0034, B0045, B0046, B0047, B0048, B0049, B0054 |
| Validation battery | B0018 |
| Test battery | B0006 |
| Feature scaling | MinMaxScaler fitted on training data |
| Optimizer | Adam for deep-learning models |
| Loss function | Mean Squared Error (MSE) |
| Maximum epochs | 100 |
| Early stopping | Validation-loss patience of 12 epochs |
| Learning-rate reduction | Factor of 0.5, patience of 5 epochs, minimum learning rate of 1e-6 |
| Batch size | 64 for deep-learning models |
| Random seed | 42 |

The sequence builder generates windows with a stride of one timestep. SOC reference values are calculated from discharge-cycle current integration using the capacity recorded in the dataset metadata for each cycle.

### Evaluation and data leakage

The split separates batteries rather than randomly distributing windows from the same battery across training, validation, and test sets. This helps evaluate performance on a battery not used for model fitting.

The feature and target scalers are fitted on the training split and then applied to the validation and test splits. Nevertheless, battery-level separation alone does not establish that every possible source of data leakage has been eliminated. Preprocessing, model selection, and the construction of reference SOC values should also be considered when interpreting the results.

---

## 📁 Data

The raw NASA Battery Dataset is **not included in this repository**.

- **Kaggle:** [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
- **NASA PCoE:** [Prognostics Center of Excellence Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

Download the dataset separately and arrange the files according to the paths expected by `src/config.py`.

---

## 📚 References

1. Saha, B., and Goebel, K. (2007). *Battery Data Set*. NASA Ames Prognostics Data Repository. [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

---

## 📄 License

The original code in this repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license text.

Third-party datasets, libraries, and other materials remain subject to their respective licenses and terms of use. The MIT License does not grant additional rights to redistribute or use those materials.

---

<div align="center">

**Ayoub HAMMAD** · Master's Thesis Project Component · 2025–2026

*Feedback and suggestions are welcome.*

</div>
