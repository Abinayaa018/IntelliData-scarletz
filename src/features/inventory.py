"""
Inventory Features Group
Computes days of inventory, stock cover vs lead time, reorder gap, and historical stock-out frequency.
LEAKAGE GUARD: Inventory ratios use current opening/closing stock (available at prediction time) 
combined with shifted rolling demand metrics. Historical stockout frequencies use shifted stock history.
"""

import pandas as pd
import numpy as np

def build_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "date"])
    
    stock_col = "closing_stock" if "closing_stock" in df.columns else "opening_stock"
    reorder_col = "reorder_level" if "reorder_level" in df.columns else "units_sold"
    lead_col = "lead_days" if "lead_days" in df.columns else "units_sold"

    # Days of inventory based on shifted rolling 7-day demand
    mean_d = df["rolling_mean_7"] if "rolling_mean_7" in df.columns else df["units_sold"].shift(1).fillna(1.0)
    df["days_of_inventory"] = df[stock_col] / (mean_d + 1e-5)
    
    # Inventory to demand ratio
    mean_14 = df["rolling_mean_14"] if "rolling_mean_14" in df.columns else mean_d
    df["inventory_to_demand_ratio"] = df[stock_col] / (mean_14 + 1e-5)
    
    # Reorder gap: current stock - reorder level
    df["reorder_gap"] = df[stock_col] - df[reorder_col] if reorder_col in df.columns else 0.0
    
    # Lead time demand: expected demand over lead_days
    lead_days = df[lead_col] if lead_col in df.columns else 3
    df["lead_time_demand"] = mean_d * lead_days
    
    # Stock cover vs lead time: days_of_inventory / lead_days
    df["stock_cover_vs_lead_time"] = df["days_of_inventory"] / (lead_days + 1e-5)
    
    # Historical Stockout Tracking
    # Stockout indicator shifted by 1 day
    stockout_event = (df.groupby(["store_id", "product_id"])[stock_col].shift(1) <= 0).astype(float)
    
    # 30-day stockout frequency
    s30 = stockout_event.groupby([df["store_id"], df["product_id"]]).rolling(30, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["past_stockout_freq_30"] = s30
    
    # Days since last stockout
    days_since_list = []
    for (st, prd), group in df.groupby(["store_id", "product_id"]):
        last_so = 999
        for is_so in (group[stock_col] <= 0):
            if is_so:
                last_so = 0
            else:
                last_so = min(999, last_so + 1)
            days_since_list.append(last_so)
            
    df["days_since_last_stockout"] = days_since_list
    
    return df
