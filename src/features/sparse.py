"""
Sparse History Fallback Module
Detects products/stores with fewer than N days of historical data (default 14 days),
imputes missing rolling/lag features using (category x store_type) averages, and sets low_history_flag.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("stocksense.features.sparse")

def handle_sparse_history(df: pd.DataFrame, min_history_days: int = 14) -> pd.DataFrame:
    df = df.copy()
    
    # 1. Count observations per store x product grain up to each date
    df["history_day_count"] = df.groupby(["store_id", "product_id"]).cumcount() + 1
    
    # 2. Flag rows with history_day_count < min_history_days
    df["low_history_flag"] = (df["history_day_count"] < min_history_days).astype(int)
    
    low_hist_count = df["low_history_flag"].sum()
    if low_hist_count > 0:
        logger.info(f"Found {low_hist_count} rows with sparse history (<{min_history_days} days). Imputing via category x store_type averages.")

        # Compute category x store_type fallback averages for numerical features
        group_cols = ["category", "store_type"] if ("category" in df.columns and "store_type" in df.columns) else ["store_id"]
        
        num_cols = [c for c in df.columns if c.startswith("rolling_") or c.startswith("lag_") or c in ["days_of_inventory", "recent_growth_rate"]]
        
        if num_cols:
            cat_store_avgs = df.groupby(group_cols)[num_cols].transform("mean")
            
            # Fill NaN values (e.g., initial lags/rolling windows) with category x store averages
            for col in num_cols:
                df[col] = df[col].fillna(cat_store_avgs[col]).fillna(0.0)

    return df
