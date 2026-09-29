"""
STOCKSENSE Data Validation Module
Ensures schema integrity, grain uniqueness, date continuity, and graceful degradation for missing columns.
"""

from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("stocksense.validation")

CORE_REQUIRED_COLUMNS = ["date", "store_id", "product_id", "units_sold"]

RECOMMENDED_COLUMNS = [
    "selling_price", "discount_pct", "promotion_flag", "opening_stock", 
    "received", "closing_stock", "reorder_level", "lead_days", "category", 
    "sub_category", "brand", "mrp", "cost_price", "shelf_life_days", 
    "store_type", "city", "floor_area_sqft", "avg_daily_customers", 
    "temp_c", "rain_mm", "holiday", "festival", "weekend", "local_event"
]

class ValidationError(Exception):
    """Raised when data validation fails critical constraints."""
    pass

class DataValidator:
    def __init__(self, config=None):
        self.config = config

    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates the input dataframe.
        Returns cleaned/cast dataframe and validation summary report.
        """
        df = df.copy()
        summary = {"is_valid": True, "warnings": [], "missing_columns": []}

        # 1. Column renaming via config mapping if provided
        if self.config:
            df = self.config.apply_column_mapping(df)

        # 2. Check Core Required Columns
        missing_core = [col for col in CORE_REQUIRED_COLUMNS if col not in df.columns]
        if missing_core:
            raise ValidationError(
                f"CRITICAL DATA VALIDATION FAILURE: Missing required core columns {missing_core}. "
                f"Available columns: {list(df.columns)}"
            )

        # 3. Check Recommended Columns for Graceful Degradation
        missing_recommended = [col for col in RECOMMENDED_COLUMNS if col not in df.columns]
        if missing_recommended:
            warning_msg = (
                f"WARNING: Dataset is missing optional/recommended columns: {missing_recommended}. "
                f"Feature engineering will degrade gracefully by skipping these groups."
            )
            logger.warning(warning_msg)
            summary["warnings"].append(warning_msg)
            summary["missing_columns"] = missing_recommended

        # 4. Cast Data Types
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception as e:
            raise ValidationError(f"Invalid date column format: {e}")

        df["store_id"] = df["store_id"].astype(str)
        df["product_id"] = df["product_id"].astype(str)
        df["units_sold"] = pd.to_numeric(df["units_sold"], errors="coerce").fillna(0.0)

        # 5. Grain Uniqueness Check: (date x store_id x product_id)
        grain_cols = ["date", "store_id", "product_id"]
        dups = df.duplicated(subset=grain_cols, keep=False)
        dup_count = dups.sum()
        if dup_count > 0:
            sample_dups = df[dups][grain_cols].head(5).to_dict("records")
            raise ValidationError(
                f"GRAIN UNIQUENESS FAILURE: Found {dup_count} duplicate rows for grain (date, store_id, product_id). "
                f"Sample duplicates: {sample_dups}"
            )

        # 6. Sort chronologically by grain
        df = df.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)

        # 7. Date Continuity Check
        gaps_found = 0
        grouped = df.groupby(["store_id", "product_id"])
        for (st, prd), group in grouped:
            diffs = group["date"].diff().dt.days
            missing_days = (diffs > 1).sum()
            if missing_days > 0:
                gaps_found += missing_days

        if gaps_found > 0:
            msg = f"Date continuity warning: Found {gaps_found} missing date gaps across store x product series."
            logger.warning(msg)
            summary["warnings"].append(msg)

        logger.info(
            f"Validation Passed: {len(df)} rows, {df['store_id'].nunique()} stores, "
            f"{df['product_id'].nunique()} products from {df['date'].min().strftime('%Y-%m-%d')} "
            f"to {df['date'].max().strftime('%Y-%m-%d')}."
        )
        return df, summary
