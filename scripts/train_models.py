"""
STOCKSENSE Master Model Training Pipeline Script
Executes end-to-end model training, benchmarking, evaluation, and artifact generation strictly from data/processed/.

Usage:
  python scripts/train_models.py
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.data_config import (
    PROCESSED_DATA_DIR,
    TRANSACTIONS_PATH,
    INVENTORY_PATH,
    PRODUCTS_PATH,
    STORES_PATH,
    EXTERNAL_FACTORS_PATH,
    MASTER_TABLE_PATH,
    MODELS_DIR,
    DEMAND_FORECAST_MODEL_DIR,
    STOCKOUT_MODEL_DIR,
    TRAINING_REPORT_PATH,
    REPORTS_DIR,
)

from src.config import load_config
from src.validation import DataValidator
from src.features.builder import FeatureBuilder
from src.targets import build_targets
from src.evaluation import chronological_split
from src.models.regression import DemandForecasterSuite
from src.models.classification import StockoutClassifierSuite
from src.recommend import RecommendationEngine

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("stocksense.train_models")


def build_master_table_if_needed() -> pd.DataFrame:
    """Builds or refreshes data/processed/master_table.csv from the 5 processed CSV files."""
    logger.info("Verifying processed source CSV files in data/processed/...")
    
    for p in [TRANSACTIONS_PATH, INVENTORY_PATH, PRODUCTS_PATH, STORES_PATH, EXTERNAL_FACTORS_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required processed dataset missing at {p}. Run python scripts/data_cleaning.py first!")

    inv_df = pd.read_csv(INVENTORY_PATH)
    tx_df = pd.read_csv(TRANSACTIONS_PATH)
    prod_df = pd.read_csv(PRODUCTS_PATH)
    stores_df = pd.read_csv(STORES_PATH)
    ext_df = pd.read_csv(EXTERNAL_FACTORS_PATH)

    inv_df["date"] = pd.to_datetime(inv_df["date"]).dt.strftime("%Y-%m-%d")
    tx_df["date"] = pd.to_datetime(tx_df["date"]).dt.strftime("%Y-%m-%d")
    ext_df["date"] = pd.to_datetime(ext_df["date"]).dt.strftime("%Y-%m-%d")

    # Aggregate transactions daily per store x product
    tx_daily = tx_df.groupby(["date", "store_id", "product_id"]).agg(
        units_sold=("quantity", "sum"),
        selling_price=("selling_price", "mean"),
        discount_pct=("discount_pct", "mean"),
        promotion_flag=("promotion_flag", "max")
    ).reset_index()

    # Merge inventory with tx_daily
    master = inv_df.merge(tx_daily, on=["date", "store_id", "product_id"], how="left")
    master["units_sold"] = master["units_sold"].fillna(master["sold"]).astype(int)
    master["promotion_flag"] = master["promotion_flag"].fillna(0).astype(int)
    master["discount_pct"] = master["discount_pct"].fillna(0.0)

    # Merge product metadata
    master = master.merge(prod_df, on="product_id", how="left")
    master["selling_price"] = master["selling_price"].fillna(master["mrp"] * (1 - master["discount_pct"] / 100.0))

    # Merge store metadata
    master = master.merge(stores_df, on="store_id", how="left")

    # Merge external factors on date & city
    master = master.merge(ext_df, on=["date", "city"], how="left")

    # Rename inventory columns to canonical names
    rename_map = {
        "opening": "opening_stock",
        "closing": "closing_stock",
        "reorder_lvl": "reorder_level"
    }
    master = master.rename(columns=rename_map)

    # Save updated master table
    master.to_csv(MASTER_TABLE_PATH, index=False)
    logger.info(f"Built master table from clean processed data: {master.shape[0]} rows x {master.shape[1]} cols saved to {MASTER_TABLE_PATH}")
    return master


def train_pipeline() -> None:
    """Executes feature engineering, dataset splitting, ML model benchmarking,

    evaluation, artifact persistence, and recommendation scoring.
    """
    print("=" * 70)
    print("        STARTING STOCKSENSE REPRODUCIBLE MODEL TRAINING        ")
    print("=" * 70)

    config = load_config()

    # 1. Load / Build Master Table from data/processed/
    df_raw = build_master_table_if_needed()

    # Convert date to datetime
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # 2. Schema Validation
    logger.info("[Step 1/6] Running schema validation...")
    validator = DataValidator(config)
    df_clean, val_summary = validator.validate(df_raw)

    # 3. Feature Engineering
    logger.info("[Step 2/6] Building zero-leakage feature matrix...")
    fb = FeatureBuilder(
        min_history_days=config.feature_params.get("sparse_history_min_days", 14),
        shelf_life_cutoff=config.feature_params.get("perishability_shelf_life_days", 7)
    )
    df_feat = fb.transform(df_clean)
    fb.generate_feature_dictionary(REPORTS_DIR / "feature_dictionary.md")

    # 4. Target Construction
    logger.info("[Step 3/6] Engineering 7-day demand & stockout targets...")
    df_target = build_targets(
        df_feat,
        horizon_days=config.target_params.get("forecast_horizon_days", 7),
        threshold_multiplier=config.target_params.get("stockout_threshold_demand_multiplier", 1.0)
    )

    # 5. Chronological Train/Val/Test Split
    logger.info("[Step 4/6] Performing chronological dataset split...")
    train_df, val_df, test_df = chronological_split(
        df_target,
        train_ratio=config.model_params.get("train_ratio", 0.70),
        val_ratio=config.model_params.get("val_ratio", 0.15)
    )
    logger.info(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 6. Train Regression Suite (Demand Forecasting)
    logger.info("[Step 5/6] Benchmarking Demand Forecasting regression models...")
    reg_suite = DemandForecasterSuite(random_seed=config.model_params.get("random_seed", 42))
    reg_benchmark_df, reg_info = reg_suite.benchmark_and_train(train_df, val_df, test_df)

    # 7. Train Classification Suite (Stock-out Prediction)
    logger.info("[Step 6/6] Benchmarking Stock-Out Prediction classification models...")
    clf_suite = StockoutClassifierSuite(
        random_seed=config.model_params.get("random_seed", 42),
        high_risk_thresh=config.model_params.get("risk_thresholds", {}).get("high", 0.70),
        med_risk_thresh=config.model_params.get("risk_thresholds", {}).get("medium", 0.40)
    )
    clf_benchmark_df, clf_info = clf_suite.benchmark_and_train(train_df, val_df, test_df)

    # 8. Save Trained Models & Reports
    logger.info("Saving trained models and performance reports...")
    
    # Save benchmark tables
    reg_benchmark_df.to_csv(REPORTS_DIR / "model_comparison_regression.csv", index=False)
    clf_benchmark_df.to_csv(REPORTS_DIR / "model_comparison_classification.csv", index=False)

    # Save joblib models in models/ and subdirectories
    joblib.dump(reg_suite.best_pipeline, MODELS_DIR / "demand_forecast_model.joblib")
    joblib.dump(clf_suite.best_calibrated_pipeline, MODELS_DIR / "stockout_risk_model.joblib")

    joblib.dump(reg_suite.best_pipeline, DEMAND_FORECAST_MODEL_DIR / "demand_forecast_model.joblib")
    joblib.dump(clf_suite.best_calibrated_pipeline, STOCKOUT_MODEL_DIR / "stockout_risk_model.joblib")

    # Build Training Report JSON
    reg_top_metrics = reg_benchmark_df.iloc[0].to_dict()
    clf_top_metrics = clf_benchmark_df.iloc[0].to_dict()

    feature_cols = [c for c in df_feat.columns if c not in ["date", "store_id", "product_id", "next_7_day_demand", "stockout_flag"]]

    training_report = {
        "project": "STOCKSENSE",
        "data_source": "data/processed/master_table.csv",
        "training_dates": {
            "start": train_df["date"].min().strftime("%Y-%m-%d"),
            "end": train_df["date"].max().strftime("%Y-%m-%d"),
        },
        "dataset_split": {
            "total_records": len(df_target),
            "train_records": len(train_df),
            "val_records": len(val_df),
            "test_records": len(test_df),
        },
        "models_trained": [
            {
                "model_name": "Demand Forecasting",
                "algorithm": reg_info["best_model"],
                "target": "next_7_day_demand",
                "features_count": len(feature_cols),
                "metrics": {
                    "MAE": float(reg_top_metrics.get("MAE", 0.0)),
                    "RMSE": float(reg_top_metrics.get("RMSE", 0.0)),
                    "sMAPE_pct": float(reg_top_metrics.get("sMAPE (%)", 0.0)),
                    "R2": float(reg_top_metrics.get("R2", 0.0)),
                },
                "saved_path": str(MODELS_DIR / "demand_forecast_model.joblib")
            },
            {
                "model_name": "Stock-out Risk Prediction",
                "algorithm": clf_info["best_model"],
                "target": "stockout_flag",
                "features_count": len(feature_cols),
                "metrics": {
                    "Accuracy": float(clf_top_metrics.get("Accuracy", 0.0)),
                    "F1_Score": float(clf_top_metrics.get("F1 Score", 0.0)),
                    "Precision": float(clf_top_metrics.get("Precision", 0.0)),
                    "Recall": float(clf_top_metrics.get("Recall", 0.0)),
                    "ROC_AUC": float(clf_top_metrics.get("ROC-AUC", 0.0)),
                },
                "saved_path": str(MODELS_DIR / "stockout_risk_model.joblib")
            }
        ],
        "feature_columns": feature_cols,
    }

    with open(TRAINING_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(training_report, f, indent=2)

    # 9. Generate Predictions & Recommendations for Latest Snapshot
    logger.info("Generating latest snapshot recommendations...")
    latest_date = df_feat["date"].max()
    latest_df = df_feat[df_feat["date"] == latest_date].copy().reset_index(drop=True)

    reg_pipe = reg_suite.best_pipeline
    clf_pipe = clf_suite.best_calibrated_pipeline

    latest_df["pred_7_day_demand"] = np.clip(reg_pipe.predict(latest_df), 0.0, None)
    latest_df["pred_stockout_prob"] = np.clip(clf_pipe.predict_proba(latest_df)[:, 1], 0.0, 1.0)

    rec_engine = RecommendationEngine(
        service_level_z=config.model_params.get("service_level_z", 1.645),
        high_risk_thresh=config.model_params.get("risk_thresholds", {}).get("high", 0.70),
        med_risk_thresh=config.model_params.get("risk_thresholds", {}).get("medium", 0.40)
    )
    rec_df = rec_engine.calculate_recommendations(latest_df)
    rec_df.to_csv(REPORTS_DIR / "latest_recommendations.csv", index=False)

    # 10. Print Console Summary Table
    print("\n" + "=" * 70)
    print("                    INTELLIDATA MODEL TRAINING                   ")
    print("=" * 70)
    print("Data Source: data/processed/master_table.csv")
    print(f"Total Records: {len(df_target):,} | Train: {len(train_df):,} | Test: {len(test_df):,}")
    print("-" * 70)
    print("MODELS TRAINED:")
    print(f"  [OK] Demand Forecasting: {reg_info['best_model']}")
    print(f"       MAE: {reg_top_metrics.get('MAE'):.2f} | RMSE: {reg_top_metrics.get('RMSE'):.2f} | R2: {reg_top_metrics.get('R2'):.3f}")
    print(f"  [OK] Stock-out Risk Prediction: {clf_info['best_model']}")
    print(f"       Accuracy: {clf_top_metrics.get('Accuracy'):.4f} | F1: {clf_top_metrics.get('F1 Score'):.4f} | ROC-AUC: {clf_top_metrics.get('ROC-AUC'):.4f}")
    print("-" * 70)
    print(f"Saved Models: {MODELS_DIR / 'demand_forecast_model.joblib'}")
    print(f"              {MODELS_DIR / 'stockout_risk_model.joblib'}")
    print(f"Saved Report: {TRAINING_REPORT_PATH}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        train_pipeline()
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        sys.exit(1)
