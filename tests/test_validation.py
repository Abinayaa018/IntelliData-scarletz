"""
Automated Unit Test: Data Validator Integrity & Schema Checks
"""

import pytest
import pandas as pd
from src.validation import DataValidator, ValidationError

def test_grain_uniqueness_failure():
    # Construct duplicate grain rows
    df_dup = pd.DataFrame([
        {"date": "2026-01-01", "store_id": "S01", "product_id": "P01", "units_sold": 10},
        {"date": "2026-01-01", "store_id": "S01", "product_id": "P01", "units_sold": 15}, # DUPLICATE GRAIN!
    ])
    
    val = DataValidator()
    with pytest.raises(ValidationError, match="GRAIN UNIQUENESS FAILURE"):
        val.validate(df_dup)

def test_missing_core_columns_failure():
    df_bad = pd.DataFrame([
        {"date": "2026-01-01", "store_id": "S01"} # Missing product_id & units_sold
    ])
    
    val = DataValidator()
    with pytest.raises(ValidationError, match="CRITICAL DATA VALIDATION FAILURE"):
        val.validate(df_bad)
