"""
STOCKSENSE Target Engineering Module
Computes 7-day future demand (regression target) and 7-day stock-out risk flag (classification target).
"""

from typing import Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("stocksense.targets")

def build_targets(
    df: pd.DataFrame, 
    horizon_days: int = 7, 
    threshold_multiplier: float = 1.0
) -> pd.DataFrame:
    """
    Computes regression target (next_7_day_demand) and classification target (stockout_flag).
    Drops trailing rows where full 7-day future horizon is unavailable.
    """
    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)
    
    # 1. Regression Target: sum of units_sold over t+1 to t+horizon_days
    # Sum of future horizon_days: shift(-1) + shift(-2) + ... + shift(-horizon_days)
    future_sums = []
    grouped = df.groupby(["store_id", "product_id"])["units_sold"]
    
    for i in range(1, horizon_days + 1):
        future_sums.append(grouped.shift(-i))
        
    df["next_7_day_demand"] = pd.concat(future_sums, axis=1).sum(axis=1, min_count=horizon_days)
    
    # 2. Classification Target: Operational Stock-Out Flag
    # Defined as 1 if:
    # (A) Closing stock falls below threshold (threshold_multiplier * rolling_mean_7 demand)
    # (B) OR projected 7-day demand > available stock (closing_stock + received)
    stock_col = "closing_stock" if "closing_stock" in df.columns else "opening_stock"
    received_col = "received" if "received" in df.columns else stock_col
    demand_col = "rolling_mean_7" if "rolling_mean_7" in df.columns else "units_sold"
    
    stock_available = df[stock_col] + (df[received_col] if received_col in df.columns else 0.0)
    min_safe_stock = threshold_multiplier * df[demand_col]
    
    insufficient_stock = df["next_7_day_demand"] > stock_available
    low_stock_threshold = df[stock_col] <= min_safe_stock
    
    df["stockout_flag"] = (insufficient_stock | low_stock_threshold).astype(int)
    
    # Set stockout_flag to NaN if next_7_day_demand is NaN (trailing boundary)
    df.loc[df["next_7_day_demand"].isna(), "stockout_flag"] = np.nan
    
    initial_len = len(df)
    valid_df = df.dropna(subset=["next_7_day_demand"]).copy()
    valid_df["stockout_flag"] = valid_df["stockout_flag"].astype(int)
    
    dropped = initial_len - len(valid_df)
    logger.info(
        f"Target engineering complete. Dropped {dropped} trailing horizon rows. "
        f"Valid target dataset: {len(valid_df)} rows. Stock-out positive rate: {valid_df['stockout_flag'].mean():.2%}"
    )
    
    return valid_df
