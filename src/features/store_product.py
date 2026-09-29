"""
Store, Product, and External Features Group
Computes perishability, store attributes, weather variations, and event indicators.
LEAKAGE GUARD: Temperature changes and weather shifts use shifted metrics.
"""

import pandas as pd
import numpy as np

def build_store_product_features(df: pd.DataFrame, shelf_life_cutoff: int = 7) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "date"])
    
    # 1. Perishability Flag
    if "shelf_life_days" in df.columns:
        df["perishability_flag"] = (df["shelf_life_days"] <= shelf_life_cutoff).astype(int)
    else:
        df["perishability_flag"] = 0

    # 2. Weather & External Shifts
    if "temp_c" in df.columns:
        temp_lag = df.groupby(["store_id", "product_id"])["temp_c"].shift(1)
        df["temp_change"] = df["temp_c"] - temp_lag.fillna(df["temp_c"])
    else:
        df["temp_c"] = 25.0
        df["temp_change"] = 0.0

    if "rain_mm" not in df.columns:
        df["rain_mm"] = 0.0

    if "local_event" not in df.columns:
        df["local_event"] = 0

    return df
