"""
Automated Unit Test: Chronological Split Integrity
Verifies strictly non-overlapping, ordered time series splits.
"""

import pytest
import pandas as pd
from src.evaluation import chronological_split

def test_chronological_split_ordering():
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates, "val": range(100)})
    
    tr, val, ts = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    
    assert tr["date"].max() < val["date"].min(), "Split Failure: Train dates overlap with Validation dates!"
    assert val["date"].max() < ts["date"].min(), "Split Failure: Validation dates overlap with Test dates!"
    assert len(tr) + len(val) + len(ts) == len(df), "Split Failure: Total rows do not match original dataset!"
