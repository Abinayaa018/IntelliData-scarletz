"""
Automated Unit Test: Decision Intelligence & Recommendation Engine Math
"""

import pytest
import pandas as pd
from src.recommend import RecommendationEngine

def test_recommendation_engine_math():
    df = pd.DataFrame([{
        "store_id": "S01",
        "product_id": "P01",
        "product_name": "Test Milk",
        "category": "Dairy",
        "closing_stock": 20.0,
        "received": 0.0,
        "pred_7_day_demand": 70.0,
        "pred_stockout_prob": 0.85,
        "selling_price": 50.0,
        "lead_days": 4,
        "reorder_level": 30.0,
        "shelf_life_days": 5
    }])
    
    engine = RecommendationEngine(service_level_z=1.645)
    rec_df = engine.calculate_recommendations(df)
    
    rec = rec_df.iloc[0]
    assert rec["Risk"] == "HIGH"
    assert rec["Recommended Order"] > 50 # 70 forecast + safety stock - 20 stock = >50
    assert rec["Estimated Lost Sales"] == 2500.0 # (70 - 20) * 50 = 2500.0
