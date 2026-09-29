"""
Lag Features Group
Computes historical demand lags per (store_id x product_id).
LEAKAGE GUARD: All lag features use .shift(lag_k) where lag_k >= 1.
Value at date t is strictly units_sold at date t-k.
"""

import pandas as pd
from typing import List

def build_lag_features(df: pd.DataFrame, lag_days: List[int] = [1, 7, 14, 28]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "date"])
    
    grouped = df.groupby(["store_id", "product_id"])["units_sold"]
    
    for k in lag_days:
        # Shift k days back: at date t, lag_k = units_sold at t-k
        df[f"lag_{k}"] = grouped.shift(k)
        
    return df
