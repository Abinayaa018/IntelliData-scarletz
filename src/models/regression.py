"""
STOCKSENSE Regression Model Benchmark Suite (7-Day Demand Forecasting)
Implements Baselines, Ridge/Lasso, Decision Tree, Random Forest, XGBoost, KNN, and SVR.
Uses scikit-learn Pipelines with strict train-set preprocessing.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

from src.evaluation import evaluate_regression

logger = logging.getLogger("stocksense.models.regression")

FEATURE_COLS_NUM = [
    "day_of_week", "weekend_flag", "month", "week_no", "day_of_month", 
    "festival_flag", "holiday_flag", "days_to_next_festival", "days_since_last_festival", "days_to_next_holiday",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_max_7", "rolling_min_7",
    "rolling_mean_14", "rolling_std_14", "rolling_mean_28",
    "recent_growth_rate", "coefficient_of_variation_14",
    "days_of_inventory", "inventory_to_demand_ratio", "reorder_gap", "lead_time_demand", "stock_cover_vs_lead_time",
    "past_stockout_freq_30", "days_since_last_stockout",
    "discount_pct", "price_to_mrp_ratio", "price_change", "promotion_flag", "promo_days_last_7", "days_since_last_promo", "historical_promo_lift",
    "perishability_flag", "temp_c", "rain_mm", "temp_change", "local_event", "low_history_flag"
]

FEATURE_COLS_CAT = ["store_type", "category", "city"]

class DemandForecasterSuite:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.fitted_models = {}
        self.best_model_name = None
        self.best_pipeline = None

    def _get_preprocessor(self, scale_num: bool = True) -> ColumnTransformer:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if scale_num else "passthrough")
        ])
        cat_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        return ColumnTransformer([
            ("num", num_transformer, FEATURE_COLS_NUM),
            ("cat", cat_transformer, [c for c in FEATURE_COLS_CAT if c])
        ])

    def benchmark_and_train(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Trains all permitted regressors and baselines, evaluates on test set, returns benchmark table."""
        y_train = train_df["next_7_day_demand"].values
        y_val = val_df["next_7_day_demand"].values
        y_test = test_df["next_7_day_demand"].values

        benchmarks = []

        # 1. Baseline 1: Previous 7-Day Mean Scaled x7
        base1_pred = (test_df["rolling_mean_7"] * 7.0).values
        b1_metrics = evaluate_regression(y_test, base1_pred, model_name="Baseline: Prev 7D Mean")
        benchmarks.append(b1_metrics)

        # 2. Baseline 2: Seasonal Naive (lag_7 x 7)
        base2_pred = (test_df["lag_7"] * 7.0).values
        b2_metrics = evaluate_regression(y_test, base2_pred, model_name="Baseline: Seasonal Naive")
        benchmarks.append(b2_metrics)

        # Permitted Model List
        models_to_train = {
            "Ridge Regression": (Ridge(alpha=10.0, random_state=self.random_seed), True),
            "Lasso Regression": (Lasso(alpha=0.1, random_state=self.random_seed), True),
            "Decision Tree": (DecisionTreeRegressor(max_depth=8, random_state=self.random_seed), False),
            "Random Forest": (RandomForestRegressor(n_estimators=100, max_depth=12, random_state=self.random_seed, n_jobs=-1), False),
            "XGBoost Regressor": (xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=self.random_seed, n_jobs=-1), False),
            "KNN Regressor": (KNeighborsRegressor(n_neighbors=7, n_jobs=-1), True),
            "SVM (SVR)": (SVR(C=1.0, epsilon=0.2), True)
        }

        best_mae = float("inf")

        for name, (regressor, scale_num) in models_to_train.items():
            logger.info(f"Training Regression Model: {name}")
            preprocessor = self._get_preprocessor(scale_num=scale_num)
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", regressor)
            ])
            
            # Fit on training set
            pipe.fit(train_df, y_train)
            
            # Predict on test set
            preds = pipe.predict(test_df)
            preds = np.clip(preds, 0.0, None) # Demand cannot be negative
            
            metrics = evaluate_regression(y_test, preds, model_name=name)
            benchmarks.append(metrics)
            
            self.fitted_models[name] = pipe
            
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                self.best_model_name = name
                self.best_pipeline = pipe

        benchmark_df = pd.DataFrame(benchmarks).sort_values(by="MAE").reset_index(drop=True)
        logger.info(f"Regression Benchmark Winner: {self.best_model_name} with MAE {best_mae:.3f}")
        
        return benchmark_df, {"best_model": self.best_model_name, "best_mae": best_mae}
