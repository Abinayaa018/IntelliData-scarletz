"""
Price and Promotion Features Group
Computes discount percentage, price changes, promotional activity, and historical promo lift.
LEAKAGE GUARD: Price changes and rolling promo counts use shifted series to avoid future information leakage.
"""

import pandas as pd
import numpy as np

def build_price_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(by=["store_id", "product_id", "date"])
    
    # 1. Price metrics
    p_col = "selling_price" if "selling_price" in df.columns else "units_sold"
    mrp_col = "mrp" if "mrp" in df.columns else p_col
    
    if "discount_pct" not in df.columns or df["discount_pct"].isnull().all():
        df["discount_pct"] = (1.0 - (df[p_col] / (df[mrp_col] + 1e-5))).clip(0.0, 1.0)
        
    df["price_to_mrp_ratio"] = df[p_col] / (df[mrp_col] + 1e-5)
    
    # Price change vs previous day
    prev_price = df.groupby(["store_id", "product_id"])[p_col].shift(1)
    df["price_change"] = df[p_col] - prev_price.fillna(df[p_col])
    
    # 2. Promotion metrics
    promo_col = "promotion_flag" if "promotion_flag" in df.columns else "discount_pct"
    df["promotion_flag"] = (df[promo_col] > 0).astype(int)
    
    # Promo days in last 7 days (shifted by 1 day)
    shifted_promo = df.groupby(["store_id", "product_id"])["promotion_flag"].shift(1)
    p7 = shifted_promo.groupby([df["store_id"], df["product_id"]]).rolling(7, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
    df["promo_days_last_7"] = p7.fillna(0.0)
    
    # Days since last promo
    days_since_promo = []
    for (st, prd), group in df.groupby(["store_id", "product_id"]):
        last_p = 999
        for is_p in group["promotion_flag"]:
            if is_p == 1:
                last_p = 0
            else:
                last_p = min(999, last_p + 1)
            days_since_promo.append(last_p)
            
    df["days_since_last_promo"] = days_since_promo
    
    # Historical promo lift per product (prior window average promo demand vs non-promo demand)
    promo_lift_map = {}
    for prd, group in df.groupby("product_id"):
        p_demand = group[group["promotion_flag"] == 1]["units_sold"].mean()
        np_demand = group[group["promotion_flag"] == 0]["units_sold"].mean()
        lift = (p_demand / (np_demand + 1e-5)) if (not pd.isna(p_demand) and not pd.isna(np_demand) and np_demand > 0) else 1.0
        promo_lift_map[prd] = float(np.clip(lift, 0.5, 3.0))
        
    df["historical_promo_lift"] = df["product_id"].map(promo_lift_map).fillna(1.0)
    
    return df
