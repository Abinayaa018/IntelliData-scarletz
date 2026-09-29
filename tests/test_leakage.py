"""
Automated Unit Test: Data Leakage Guard Audit
Verifies that lag and rolling features at date t do NOT contain information from date >= t.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureBuilder

def test_lag_and_rolling_zero_leakage():
    # Construct a synthetic time series with a huge demand spike on day 10
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store_id": "S01",
        "product_id": "P01",
        "units_sold": [10.0] * 9 + [1000.0] + [10.0] * 10, # day 10 spike
        "closing_stock": 50.0,
        "opening_stock": 50.0,
        "reorder_level": 20.0,
        "lead_days": 3,
        "mrp": 100.0,
        "selling_price": 100.0
    })
    
    fb = FeatureBuilder()
    df_feat = fb.transform(df)
    
    # At day 10 (index 9), lag_1 must equal 10.0 (day 9 demand), NOT 1000.0!
    assert df_feat.iloc[9]["lag_1"] == 10.0, "Leakage Failure: lag_1 on spike day contains spike value!"
    
    # At day 10 (index 9), rolling_mean_7 must equal 10.0, NOT including day 10's 1000.0!
    assert df_feat.iloc[9]["rolling_mean_7"] == 10.0, "Leakage Failure: rolling_mean_7 at date t includes date t demand!"
    
    # Day 11 (index 10) is the first day lag_1 should reflect the day 10 spike
    assert df_feat.iloc[10]["lag_1"] == 1000.0
