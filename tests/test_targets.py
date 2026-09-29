"""
Automated Unit Test: Target Construction & Trailing Horizon Drop Logic
"""

import pytest
import pandas as pd
import numpy as np
from src.targets import build_targets

def test_target_construction_and_drop_rules():
    dates = pd.date_range("2026-01-01", periods=14, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store_id": "S01",
        "product_id": "P01",
        "units_sold": [10.0] * 14,
        "closing_stock": [100.0] * 14,
        "received": [0.0] * 14,
        "rolling_mean_7": [10.0] * 14
    })
    
    # 7-day future horizon target calculation
    df_target = build_targets(df, horizon_days=7)
    
    # Should drop last 7 days because full 7-day future is unavailable
    assert len(df_target) == 7
    assert df_target.iloc[0]["next_7_day_demand"] == 70.0 # 7 days * 10 units
