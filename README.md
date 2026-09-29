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
                               | - Landing Page Cinematic 3D Hero      |
                               | - Sidebar Stepper & 7 Task Pages      |
                               | - Three.js 3D Store Digital Twin      |
                               +---------------------------------------+
```

---

## 🎨 Guided App Flow & Visual Identity

The STOCKSENSE dashboard follows a single continuous guided journey:

1. **Cinematic 3D Landing Page**:
   - Full-screen holographic supermarket animation built in Three.js (`r128`).
   - One-line tagline (*"Predict demand. Prevent stock-outs. Power better decisions."*).
   - Primary **🚀 ENTER COMMAND CENTRE** button (session state routing).
2. **Sidebar Control Panel**:
   - **Guided Stepper**: `1 Load Data -> 2 Run Prediction -> 3 Review Risk -> 4 Take Action`.
   - **Data Source Selector**: Choice between "Use demo data" and custom `master_table.csv` upload.
   - **5-Stage Pipeline Execution Button**: Feature Engineering -> Demand Forecast -> Stock-Out Risk -> Explanations -> Recommendations with stage progress bar and timestamping.
   - **Global Multi-Select Filters**: Store, City, Category, Risk level, Product search, Reorder toggle, and **🔄 Reset All Filters**.
3. **7 Task Dashboard Pages**:
   - `🌐 Executive Summary`: Equal-height KPI grid, 540px Three.js 3D Store Twin with warning beacons and 7-day drain replay.
   - `📈 Demand Intelligence`: Actual vs forecast charts with confidence bands and category trends.
   - `🛡️ Inventory Risk`: Risk tiles, probability heatmap, and sortable risk matrix.
   - `🛒 Manager Action Centre`: Prioritized action cards with order approval toggles and lost sales prevented counter.
   - `🧠 Explainability`: Global permutation importance and local plain-language driver breakdowns.
   - `🎛️ What-If Lab`: Interactive discount, supplier delay, and festival uplift simulator.
   - `🔬 Model Performance`: Regression & classification benchmark comparison tables and model justification.

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Demo Mode Pipeline
To generate synthetic master table data and run model training + prediction CLI:
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
- **Rolling Features**: All rolling statistics apply `.shift(1)` **BEFORE** `.rolling(window)`. At date \(t\), rolling windows cover \([t-w, t-1]\) and NEVER include date \(t\)'s demand.
- **Preprocessing**: Encoders and scalers are fitted strictly on training dates inside scikit-learn `Pipeline` objects.

Full feature dictionary: [`reports/feature_dictionary.md`](file:///d:/IntelliData-scarletz/reports/feature_dictionary.md).

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
