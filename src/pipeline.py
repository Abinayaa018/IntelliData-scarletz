"""
STOCKSENSE Master Pipeline CLI Module
Entry points:
  - python -m src.pipeline train   : Feature engineering, model benchmarking, calibration, artifact saving
  - python -m src.pipeline predict : Scoring latest snapshot & generating recommendation CSV
  - python -m src.pipeline demo    : Synthetic data generation + full end-to-end execution
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import logging

from src.config import load_config
from src.validation import DataValidator
from src.synthetic_data import generate_synthetic_master_table
from src.features.builder import FeatureBuilder
from src.targets import build_targets
from src.evaluation import chronological_split
from src.models.regression import DemandForecasterSuite
from src.models.classification import StockoutClassifierSuite
from src.explain import ModelExplainer
from src.recommend import RecommendationEngine

logger = logging.getLogger("stocksense.pipeline")

def run_train(config_path: str = None) -> None:
    """Executes full model training, tuning, calibration, and artifact persistence."""
    logger.info("================ STARTING STOCKSENSE TRAINING PIPELINE ================")
    config = load_config(config_path)
    
    # 1. Load Data
    master_path = Path(config.raw_config.get("data", {}).get("master_table_path", "data/processed/master_table.csv"))
    if not master_path.exists():
        synth_path = Path(config.raw_config.get("data", {}).get("synthetic_table_path", "data/processed/master_table_synthetic.csv"))
        if synth_path.exists():
            logger.warning(f"Master table {master_path} not found. Using synthetic dataset at {synth_path}.")
            master_path = synth_path
        else:
            logger.info("No existing dataset found. Generating synthetic dataset for demo training...")
            df_synth = generate_synthetic_master_table(num_days=180, output_path=synth_path)
            master_path = synth_path
            
    df_raw = pd.read_csv(master_path)
    
    # 2. Validation
    validator = DataValidator(config)
    df_clean, val_summary = validator.validate(df_raw)
    
    # 3. Feature Engineering
    fb = FeatureBuilder(
        min_history_days=config.feature_params.get("sparse_history_min_days", 14),
        shelf_life_cutoff=config.feature_params.get("perishability_shelf_life_days", 7)
    )
    df_feat = fb.transform(df_clean)
    
    # Generate Feature Dictionary
    reports_dir = config.root_dir / "reports"
    fb.generate_feature_dictionary(reports_dir / "feature_dictionary.md")
    
    # 4. Target Construction
    df_target = build_targets(
        df_feat, 
        horizon_days=config.target_params.get("forecast_horizon_days", 7),
        threshold_multiplier=config.target_params.get("stockout_threshold_demand_multiplier", 1.0)
    )
    
    # 5. Chronological Split
    train_df, val_df, test_df = chronological_split(
        df_target,
        train_ratio=config.model_params.get("train_ratio", 0.70),
        val_ratio=config.model_params.get("val_ratio", 0.15)
    )
    
    # 6. Train Regression Suite (Model 1)
    reg_suite = DemandForecasterSuite(random_seed=config.model_params.get("random_seed", 42))
    reg_benchmark_df, reg_info = reg_suite.benchmark_and_train(train_df, val_df, test_df)
    
    # 7. Train Classification Suite (Model 2)
    clf_suite = StockoutClassifierSuite(
        random_seed=config.model_params.get("random_seed", 42),
        high_risk_thresh=config.model_params.get("risk_thresholds", {}).get("high", 0.70),
        med_risk_thresh=config.model_params.get("risk_thresholds", {}).get("medium", 0.40)
    )
    clf_benchmark_df, clf_info = clf_suite.benchmark_and_train(train_df, val_df, test_df)
    
    # 8. Save Artifacts & Reports
    models_dir = config.root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Model Comparison Tables
    reg_benchmark_df.to_csv(reports_dir / "model_comparison_regression.csv", index=False)
    clf_benchmark_df.to_csv(reports_dir / "model_comparison_classification.csv", index=False)
    
    # Save Pipelines with Joblib
    joblib.dump(reg_suite.best_pipeline, models_dir / "demand_forecast_model.joblib")
    joblib.dump(clf_suite.best_calibrated_pipeline, models_dir / "stockout_risk_model.joblib")
    
    # Generate Model Card JSON
    model_card = {
        "project": "STOCKSENSE",
        "training_dates": {
            "start": train_df["date"].min().strftime("%Y-%m-%d"),
            "end": train_df["date"].max().strftime("%Y-%m-%d"),
            "row_count": len(train_df)
        },
        "regression_winner": reg_info["best_model"],
        "regression_metrics": reg_benchmark_df.iloc[0].to_dict(),
        "classification_winner": clf_info["best_model"],
        "classification_metrics": clf_benchmark_df.iloc[0].to_dict(),
        "risk_thresholds": config.model_params.get("risk_thresholds", {"high": 0.70, "medium": 0.40})
    }
    
    with open(models_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    # Model Justification Report
    justification_content = f"""# STOCKSENSE Model Selection & Business Justification Report

## 1. Demand Forecasting (Regression)
- **Selected Winning Model**: `{reg_info['best_model']}`
- **Test Set Performance**: MAE = {reg_benchmark_df.iloc[0]['MAE']}, RMSE = {reg_benchmark_df.iloc[0]['RMSE']}, sMAPE = {reg_benchmark_df.iloc[0]['sMAPE (%)']}%, R² = {reg_benchmark_df.iloc[0]['R2']}.
- **Business Justification**: Retail demand forecasting is asymmetrical — under-forecasting leads to direct unfulfilled demand and lost store revenue, whereas slight over-forecasting is buffered by safety stock. The selected model minimizes both absolute demand error (MAE) and under-forecast rate ({reg_benchmark_df.iloc[0]['Under-Forecast Rate (%)']}%).

## 2. Stock-Out Risk Prediction (Classification)
- **Selected Winning Model**: `{clf_info['best_model']}` (Calibrated via Platt Scaling)
- **Test Set Performance**: Accuracy = {clf_benchmark_df.iloc[0]['Accuracy']}, F1 = {clf_benchmark_df.iloc[0]['F1 Score']}, ROC-AUC = {clf_benchmark_df.iloc[0]['ROC-AUC']}.
- **Business Justification**: A missed stock-out (false negative) causes immediate revenue loss and customer dissatisfaction. Calibrated probability outputs ensure risk percentages reflect true empirical likelihood.
"""
    with open(reports_dir / "model_justification.md", "w", encoding="utf-8") as f:
        f.write(justification_content)

    logger.info("================ TRAINING PIPELINE COMPLETED SUCCESSFULLY ================")

def run_predict(config_path: str = None) -> pd.DataFrame:
    """Loads latest snapshot, predicts 7-day demand & stockout probability, generates recommendations."""
    logger.info("================ STARTING STOCKSENSE PREDICTION PIPELINE ================")
    config = load_config(config_path)
    
    models_dir = config.root_dir / "models"
    reg_pipe = joblib.load(models_dir / "demand_forecast_model.joblib")
    clf_pipe = joblib.load(models_dir / "stockout_risk_model.joblib")
    
    # Load Master Table
    synth_path = config.root_dir / "data" / "processed" / "master_table_synthetic.csv"
    real_path = config.root_dir / "data" / "processed" / "master_table.csv"
    master_path = real_path if real_path.exists() else synth_path
    
    df_raw = pd.read_csv(master_path)
    validator = DataValidator(config)
    df_clean, _ = validator.validate(df_raw)
    
    fb = FeatureBuilder()
    df_feat = fb.transform(df_clean)
    
    # Select latest available snapshot date per store x product
    latest_date = df_feat["date"].max()
    logger.info(f"Scoring latest available date snapshot: {latest_date.strftime('%Y-%m-%d')}")
    latest_df = df_feat[df_feat["date"] == latest_date].copy().reset_index(drop=True)
    
    # Run Predictions
    latest_df["pred_7_day_demand"] = np.clip(reg_pipe.predict(latest_df), 0.0, None)
    latest_df["pred_stockout_prob"] = np.clip(clf_pipe.predict_proba(latest_df)[:, 1], 0.0, 1.0)
    
    # Run Recommendation Engine
    engine = RecommendationEngine(
        service_level_z=config.model_params.get("service_level_z", 1.645),
        high_risk_thresh=config.model_params.get("risk_thresholds", {}).get("high", 0.70),
        med_risk_thresh=config.model_params.get("risk_thresholds", {}).get("medium", 0.40)
    )
    rec_df = engine.calculate_recommendations(latest_df)
    
    reports_dir = config.root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(reports_dir / "latest_recommendations.csv", index=False)
    
    logger.info(f"Prediction complete. Generated {len(rec_df)} actionable store recommendations at {reports_dir / 'latest_recommendations.csv'}")
    return rec_df

def run_demo(config_path: str = None) -> None:
    """Generates synthetic dataset and runs full training + prediction end-to-end."""
    logger.info("================ RUNNING DEMO MODE (SYNTHETIC DATA) ================")
    config = load_config(config_path)
    synth_path = config.root_dir / "data" / "processed" / "master_table_synthetic.csv"
    generate_synthetic_master_table(num_days=180, output_path=synth_path)
    run_train(config_path)
    run_predict(config_path)
    logger.info("================ DEMO MODE COMPLETED SUCCESSFULLY ================")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "train":
            run_train()
        elif cmd == "predict":
            run_predict()
        elif cmd == "demo":
            run_demo()
        else:
            print(f"Unknown command '{cmd}'. Usage: python -m src.pipeline [train|predict|demo]")
    else:
        run_demo()
