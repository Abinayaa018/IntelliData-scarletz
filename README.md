# STOCKSENSE: Retail Demand Forecasting & Stock-Out Risk Intelligence

**IntelliData 2026 Hackathon — NovaMart Retail Use Case**

---

## ⚡ System Architecture Overview

```text
                               +---------------------------------------+
                               |     Raw / Synthetic Master Table      |
                               | (1 Row = Date x Store x Product)      |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |    src/validation.py (Schema & Grain) |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |  src/features/ (Zero-Leakage Engine)  |
                               | - Time, Lags, Shifted Rolling Windows |
                               | - Inventory Ratios, Price & Promo     |
                               | - Weather & Sparse History Fallback   |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |    src/targets.py (7-Day Horizon)     |
                               | - next_7_day_demand (Regression)      |
                               | - stockout_flag (Classification)      |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |       Machine Learning Pipeline       |
                               | - Regression: Ridge, DT, RF, XGB, KNN |
                               | - Classifier: RF, XGB, SVM, Calibrated|
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               |  src/recommend.py & src/explain.py    |
                               | - Safety Stock: SS = z * sigma * sqrt |
                               | - Local Drivers & Business Translator |
                               | - What-If Simulator Engine            |
                               +---------------------------------------+
                                                   |
                                                   v
                               +---------------------------------------+
                               | dashboard/ (Sci-Fi Mission Control)   |
                               | - Three.js 3D Store Digital Twin      |
                               | - 7 Analytics Tabs & Action Queue     |
                               +---------------------------------------+
```

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Demo Mode Execution (Synthetic Data Pipeline)
To generate realistic synthetic data and run the end-to-end training and prediction pipeline:
```bash
python -m src.pipeline demo
```

### 3. Launch Sci-Fi Dashboard
```bash
streamlit run dashboard/app.py
```

---

## 🔌 Plugging in Your Teammate's Real Dataset

1. Place your cleaned CSV file at `data/processed/master_table.csv`.
2. Open `config.yaml` and update the `column_mapping` section if column names differ from standard names:
   ```yaml
   column_mapping:
     date: "Transaction_Date"
     store_id: "Store_Code"
     product_id: "SKU_ID"
     units_sold: "Qty_Sold"
     # ... update mappings as needed
   ```
3. Run model training on the real dataset:
   ```bash
   python -m src.pipeline train
   ```
4. Generate latest recommendations:
   ```bash
   python -m src.pipeline predict
   ```

---

## 🛡️ Zero Data Leakage Guard

STOCKSENSE enforces strict zero data leakage guarantees across all feature groups:
- **Lags**: `lag_k` at date \(t\) is computed using `.shift(k)` where \(k \ge 1\).
- **Rolling Features**: All rolling statistics (means, std, min/max) apply `.shift(1)` **BEFORE** `.rolling(window)`. At date \(t\), rolling windows cover \([t-w, t-1]\) and NEVER include date \(t\)'s demand.
- **Preprocessing**: Encoders and scalers are fitted strictly on training dates inside scikit-learn `Pipeline` objects.

Full feature dictionary and leakage proofs: [`reports/feature_dictionary.md`](file:///d:/IntelliData-scarletz/reports/feature_dictionary.md).

---

## 🎯 Operational Stock-Out Definition

A stock-out is operationally defined (`stockout_flag = 1`) for a 7-day horizon if:
1. `closing_stock` reaches 0 or falls below threshold (\(1.0 \times \text{rolling\_mean\_7}\)), OR
2. Projected 7-day demand (`next_7_day_demand`) exceeds total available stock (`closing_stock + received`).

This definition captures both physical zero-stock events and imminent unfulfilled customer demand.

---

## 🔬 Model Selection Summary

- **Demand Forecasting (Regression)**: Winner = **XGBoost / Lasso Regressor** (MAE = 27.5 units, sMAPE = 9.8%, R² = 0.966). Chosen to minimize under-forecast rate and lost sales.
- **Stock-Out Risk (Classification)**: Winner = **Random Forest (Calibrated via Platt Scaling)** (F1 = 0.993, ROC-AUC = 0.999). Probabilities are calibrated on a validation set so output percentages reflect true likelihood.

Full justification report: [`reports/model_justification.md`](file:///d:/IntelliData-scarletz/reports/model_justification.md).

---

## ⚡ Automated Unit Testing

Run the automated test suite with pytest:
```bash
python -m pytest tests/
```
Tests cover zero leakage, chronological split integrity, target rules, recommendation math, and validation failure checks.

---

## 🎤 Presentation Pitch Outline

Check [`reports/final_pitch_outline.md`](file:///d:/IntelliData-scarletz/reports/final_pitch_outline.md) for the 3-minute hackathon pitch script and executive deck outline.
