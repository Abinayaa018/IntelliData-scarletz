"""
STOCKSENSE Central Data Configuration Module
Defines single source of truth paths and dynamic data access specifications for all backend, ML, and dashboard components.
"""

import os
from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Processed Dataset Files (SINGLE SOURCE OF TRUTH)
TRANSACTIONS_PATH = PROCESSED_DATA_DIR / "transactions.csv"
INVENTORY_PATH = PROCESSED_DATA_DIR / "inventory.csv"
PRODUCTS_PATH = PROCESSED_DATA_DIR / "products.csv"
STORES_PATH = PROCESSED_DATA_DIR / "stores.csv"
EXTERNAL_FACTORS_PATH = PROCESSED_DATA_DIR / "external_factors.csv"
MASTER_TABLE_PATH = PROCESSED_DATA_DIR / "master_table.csv"

# Preprocessing & Data Quality Reports
PROFILING_REPORT_CSV = PROCESSED_DATA_DIR / "data_profiling_report.csv"
PROFILING_REPORT_TXT = PROCESSED_DATA_DIR / "data_profiling_report.txt"
OUTLIER_REPORT_PATH = PROCESSED_DATA_DIR / "outlier_report.csv"
CLEANING_LOG_PATH = PROCESSED_DATA_DIR / "cleaning_log.csv"
QUALITY_REPORT_PATH = PROCESSED_DATA_DIR / "data_quality_report.csv"

# Backend Analysis Reports
DATASET_ANALYSIS_REPORT_JSON = PROCESSED_DATA_DIR / "dataset_analysis_report.json"
DATASET_ANALYSIS_REPORT_CSV = PROCESSED_DATA_DIR / "dataset_analysis_report.csv"

# Models & Reports Directories
MODELS_DIR = PROJECT_ROOT / "models"
DEMAND_FORECAST_MODEL_DIR = MODELS_DIR / "demand_forecasting"
STOCKOUT_MODEL_DIR = MODELS_DIR / "stockout"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.json"

# Standard Reports Directory
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure Output Directories Exist
for d in [PROCESSED_DATA_DIR, MODELS_DIR, DEMAND_FORECAST_MODEL_DIR, STOCKOUT_MODEL_DIR, REPORTS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Export Configuration Summary
def get_data_paths() -> dict:
    return {
        "processed_dir": str(PROCESSED_DATA_DIR),
        "transactions": str(TRANSACTIONS_PATH),
        "inventory": str(INVENTORY_PATH),
        "products": str(PRODUCTS_PATH),
        "stores": str(STORES_PATH),
        "external_factors": str(EXTERNAL_FACTORS_PATH),
        "master_table": str(MASTER_TABLE_PATH),
    }
