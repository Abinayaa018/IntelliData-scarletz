# STOCKSENSE Model Selection & Business Justification Report

## 1. Demand Forecasting (Regression)
- **Selected Winning Model**: `XGBoost Regressor`
- **Test Set Performance**: MAE = 26.247, RMSE = 40.227, sMAPE = 9.57%, R² = 0.9695.
- **Business Justification**: Retail demand forecasting is asymmetrical — under-forecasting leads to direct unfulfilled demand and lost store revenue, whereas slight over-forecasting is buffered by safety stock. The selected model minimizes both absolute demand error (MAE) and under-forecast rate (42.47%).

## 2. Stock-Out Risk Prediction (Classification)
- **Selected Winning Model**: `Random Forest (Balanced)` (Calibrated via Platt Scaling)
- **Test Set Performance**: Accuracy = 0.9852, F1 = 0.9913, ROC-AUC = 0.9966.
- **Business Justification**: A missed stock-out (false negative) causes immediate revenue loss and customer dissatisfaction. Calibrated probability outputs ensure risk percentages reflect true empirical likelihood.
