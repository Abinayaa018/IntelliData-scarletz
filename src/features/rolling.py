"""
Rolling Features Group
Computes moving averages, standard deviations, min/max, growth rate, and coefficient of variation.
LEAKAGE GUARD: All rolling operations apply .shift(1) BEFORE .rolling(window).
At date t, the window spans [t-window, t-1] and NEVER includes units_sold at date t.
"""

import pandas as pd
import numpy as np

def build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "date"])
    
    # Apply shift(1) first to exclude current day demand from rolling metrics
    shifted = df.groupby(["store_id", "product_id"])["units_sold"].shift(1)
    
    # Rolling stats for window 7
    r7 = shifted.groupby([df["store_id"], df["product_id"]]).rolling(7, min_periods=1)
    df["rolling_mean_7"] = r7.mean().reset_index(level=[0, 1], drop=True)
    df["rolling_std_7"] = r7.std().fillna(0.0).reset_index(level=[0, 1], drop=True)
    df["rolling_max_7"] = r7.max().reset_index(level=[0, 1], drop=True)
    df["rolling_min_7"] = r7.min().reset_index(level=[0, 1], drop=True)
    
    # Rolling stats for window 14
    r14 = shifted.groupby([df["store_id"], df["product_id"]]).rolling(14, min_periods=1)
    df["rolling_mean_14"] = r14.mean().reset_index(level=[0, 1], drop=True)
    df["rolling_std_14"] = r14.std().fillna(0.0).reset_index(level=[0, 1], drop=True)
    
    # Rolling stats for window 28
    r28 = shifted.groupby([df["store_id"], df["product_id"]]).rolling(28, min_periods=1)
    df["rolling_mean_28"] = r28.mean().reset_index(level=[0, 1], drop=True)
    
    # Recent Growth Rate: 7-day vs prior 7-day (lagged by 7 days)
    shifted_7d_ago = df.groupby(["store_id", "product_id"])["rolling_mean_7"].shift(7)
    df["recent_growth_rate"] = df["rolling_mean_7"] / (shifted_7d_ago + 1e-5)
    df["recent_growth_rate"] = df["recent_growth_rate"].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Coefficient of Variation (14-day): std / (mean + eps)
    df["coefficient_of_variation_14"] = df["rolling_std_14"] / (df["rolling_mean_14"] + 1e-5)
    df["coefficient_of_variation_14"] = df["coefficient_of_variation_14"].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    
    return df
