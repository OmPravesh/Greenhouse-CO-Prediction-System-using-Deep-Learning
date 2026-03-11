# Greenhouse-CO-Prediction-System-using-Deep-Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GDTBjojGMrXzJGBtLagfIp5v97C4xR3z?usp=sharing)

---

This project implements multiple deep learning architectures to predict greenhouse CO₂ levels using IoT sensor data collected at 10-minute intervals. The models learn temporal patterns from environmental features such as temperature, humidity, light intensity, and VOC gas concentration.

---

## Overview

This project builds and compares five deep learning models to predict **CO2 levels (ppm)** inside a greenhouse one step ahead (10 minutes into the future), using 8 hours of historical sensor context. The best-performing model — **CNN-BiLSTM with Multi-Head Attention** — combines local pattern extraction, bidirectional temporal modelling, and dynamic time-step weighting.

---

## Problem Statement

> Given the last **48 timesteps** (8 hours) of 11 sensor features, predict the CO2 concentration at the next 10-minute mark.

```
Input:  X = [x(t-47), x(t-46), ..., x(t)]   shape: (48, 11)
Output: y = CO2 concentration in ppm at t+1
```

---

## Dataset

**Source:** [Greenhouse Sensor Data — 10-Minute Interval](https://www.kaggle.com/datasets/marcelboonman/greenhouse-sensor-data-10-minute-interval) by Marcel Boonman on Kaggle.

| Feature | Description |
|---|---|
| `greenhous_temperature_celsius` | Inside greenhouse temperature (°C) |
| `greenhouse_humidity_percentage` | Inside relative humidity (%) |
| `greenhouse_illuminance_lux` | Light intensity inside (lux) |
| `online_temperature_celsius` | Outside ambient temperature (°C) |
| `online_humidity_percentage` | Outside relative humidity (%) |
| `greenhouse_total_volatile_organic_compounds_ppb` | TVOC inside greenhouse (ppb) |
| `greenhouse_equivalent_co2_ppm` | **Target — CO2 concentration (ppm)** |

---

## Engineered Features

Three additional features are computed before training:

| Feature | Formula | Purpose |
|---|---|---|
| `hour_sin` | `sin(2π × hour / 24)` | Cyclical time encoding |
| `hour_cos` | `cos(2π × hour / 24)` | Cyclical time encoding |
| `tvoc_roll_mean` | 12-step rolling mean of TVOC | Short-term TVOC trend |
| `tvoc_roll_std` | 12-step rolling std of TVOC | TVOC volatility signal |
| `co2_roll_mean` | 6-step rolling mean of CO2 | CO2 lag context |

> **Why cyclical encoding?** Raw hour integers (0–23) treat hour 23 and hour 0 as 23 units apart, but they are only 1 hour apart in reality. Sine/cosine encoding maps both to adjacent points on a unit circle, correctly representing their temporal proximity.

---

## Models Compared

| # | Model | Architecture |
|---|---|---|
| 1 | **LSTM** | Stacked LSTM (64→32 units) |
| 2 | **CNN Multi-scale** | Parallel Conv1D with kernel sizes 3, 5, 7 |
| 3 | **CNN-LSTM** | Conv1D feature extraction → LSTM temporal modelling |
| 4 | **CNN-BiLSTM + Attention** | Conv1D → Bidirectional LSTM → Multi-Head Attention *(best)* |
| 5 | **Transformer** | 2-block Transformer Encoder with positional projection |

---

## Best Model Architecture

```
Input (48, 11)
    │
    ▼
Conv1D(64, kernel=3, padding=same, ReLU, L2=1e-4)
    │
MaxPooling1D(pool=2)
    │
Dropout(0.2)
    │
Bidirectional LSTM(64, return_sequences=True)
    │
Dropout(0.2)
    │
MultiHeadAttention(heads=4, key_dim=32)  ──┐
    │                                      │  Residual
    └──────────────────────────────────────┘
    │
LayerNormalization
    │
GlobalAveragePooling1D
    │
Dense(32, ReLU, L2=1e-4)
    │
Dense(1, linear)  ──► Predicted CO2 (ppm)
```

**Why this combination works:**
- **Conv1D** — detects local short-term patterns (e.g. sudden TVOC spikes) across 3 consecutive timesteps
- **Bidirectional LSTM** — captures long-range dependencies in both forward and backward directions within the 8-hour window
- **Multi-Head Attention** — dynamically weights the most relevant timesteps for the final prediction, with 4 heads attending to different aspects simultaneously
- **Residual + LayerNorm** — stabilises gradient flow during backpropagation

---

## Pipeline

```
Cell 1  — Imports & reproducibility seeds
Cell 2  — Load & clean dataset (delimiter detection, type casting, interpolation)
Cell 3  — Validate dataframe
Cell 4  — Feature engineering (cyclical encoding, rolling stats)
Cell 5  — EDA plots (correlation heatmap, CO2 over time, distribution, TVOC scatter)
Cell 6  — Train/test split (80/20 chronological) + MinMax scaling
Cell 7  — Sliding window sequence creation (SEQ_LEN=48)
Cell 8  — Training callbacks (EarlyStopping + ReduceLROnPlateau)
Cell 9  — Model 1: Stacked LSTM
Cell 10 — Model 2: Multi-scale CNN
Cell 11 — Model 3: CNN-LSTM
Cell 12 — Model 4: CNN-BiLSTM + Multi-Head Attention
Cell 13 — Model 5: Transformer Encoder
Cell 14 — Evaluate all models (MAE, RMSE, R², MAPE)
Cell 15 — Comparison plots (predictions, loss curves, R² bar, MAE before/after)
```

---

## Training Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| Sequence length | 48 steps | 8 hours of context |
| Train / Test split | 80% / 20% | Chronological — no shuffling |
| Epochs (max) | 100 | EarlyStopping halts early in practice |
| Batch size | 32 | Balances stability and speed |
| Optimizer | Adam (lr=0.001) | Adaptive per-parameter learning rates |
| Loss function | MSE | Penalises large errors more heavily |
| L2 regularisation | 1e-4 | Applied to Conv and Dense layers |
| Dropout | 0.2 | Applied after Conv and LSTM blocks |
| EarlyStopping patience | 8 epochs | Restores best weights automatically |
| ReduceLROnPlateau | factor=0.5, patience=4 | Min LR floor = 1e-6 |

---

## Evaluation Metrics

All metrics are computed on the **test set** after inverse-transforming predictions back to the original ppm scale.

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | `mean(\|actual - pred\|)` | Average error in ppm |
| RMSE | `sqrt(mean((actual - pred)²))` | Penalises large errors more than MAE |
| R² | `1 - SS_res / SS_tot` | Proportion of variance explained (1.0 = perfect) |
| MAPE | `mean(\|actual - pred\| / \|actual\|) × 100` | Scale-independent percentage error |

---

## Project Structure

```
greenhouse-co2-prediction/
│
├── greenhouse_co2_prediction.ipynb   # Main Kaggle notebook
├── README.md                         # This file
```

---

## Requirements

```
tensorflow >= 2.x
numpy
pandas
matplotlib
seaborn
scikit-learn
kagglehub
```

Install via:
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kagglehub
```

---

## How to Run

1. Open the notebook on **Kaggle** (recommended — dataset downloads automatically via `kagglehub`)
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Run all cells in order from Cell 1 to Cell 15
4. Training takes approximately **5–15 minutes** depending on GPU availability and early stopping

---

## Key Results

The CNN-BiLSTM + Attention model consistently outperforms all other architectures across all four metrics, particularly on R² score (proportion of CO2 variance explained) and MAPE (percentage error), demonstrating that combining local convolution, bidirectional recurrence, and attention-based weighting is the most effective approach for this greenhouse sensor forecasting task.

---

## Notes

- **Data leakage prevention:** MinMaxScaler is fit **only** on training data; the same learned min/max is applied to the test set
- **No data shuffling:** Time-series integrity is preserved throughout — splits are strictly chronological
- **CUDA warnings on Colab/Kaggle:** Messages like `Unable to register cuFFT factory` are harmless TensorFlow initialisation logs and do not affect training

---

## License

This project uses the [Greenhouse Sensor Dataset](https://www.kaggle.com/datasets/marcelboonman/greenhouse-sensor-data-10-minute-interval) available on Kaggle. Please refer to the dataset's original license for usage terms.
