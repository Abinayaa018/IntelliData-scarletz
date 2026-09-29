"""
STOCKSENSE Synthetic Master Table Generator (Demo Mode)
Generates realistic multi-store, multi-product retail time series data with seasonality, 
promotions, weather shifts, festival spikes, and authentic stock-out occurrences.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("stocksense.synthetic")

STORES = [
    {"store_id": "S01", "store_type": "Express", "city": "Mumbai", "floor_area_sqft": 3500, "avg_daily_customers": 1200, "multiplier": 0.8},
    {"store_id": "S02", "store_type": "Supermarket", "city": "Delhi", "floor_area_sqft": 15000, "avg_daily_customers": 4500, "multiplier": 1.5},
    {"store_id": "S03", "store_type": "Hypermarket", "city": "Bengaluru", "floor_area_sqft": 45000, "avg_daily_customers": 9000, "multiplier": 2.5},
]

PRODUCTS = [
    {"product_id": "P001", "name": "Fresh Organic Milk 1L", "category": "Dairy", "sub_category": "Milk", "brand": "NutriDairy", "mrp": 60.0, "cost_price": 45.0, "base_demand": 40, "shelf_life_days": 4, "lead_days": 2, "reorder_level": 50},
    {"product_id": "P002", "name": "Artisanal White Bread 400g", "category": "Bakery", "sub_category": "Bread", "brand": "DailyBake", "mrp": 45.0, "cost_price": 30.0, "base_demand": 30, "shelf_life_days": 5, "lead_days": 2, "reorder_level": 40},
    {"product_id": "P003", "name": "Sparkling Soda 500ml", "category": "Beverages", "sub_category": "Cold Drinks", "brand": "FizzPop", "mrp": 35.0, "cost_price": 20.0, "base_demand": 55, "shelf_life_days": 180, "lead_days": 3, "reorder_level": 70},
    {"product_id": "P004", "name": "Basmati Premium Rice 5kg", "category": "Pantry", "sub_category": "Grains", "brand": "RoyalGrains", "mrp": 450.0, "cost_price": 320.0, "base_demand": 15, "shelf_life_days": 365, "lead_days": 5, "reorder_level": 30},
    {"product_id": "P005", "name": "Herbal Shampoo 250ml", "category": "Personal Care", "sub_category": "Haircare", "brand": "Botanica", "mrp": 220.0, "cost_price": 140.0, "base_demand": 12, "shelf_life_days": 730, "lead_days": 4, "reorder_level": 25},
]

FESTIVAL_DATES = ["2026-01-26", "2026-03-08", "2026-03-25", "2026-05-01", "2026-06-15"]
HOLIDAY_DATES = ["2026-01-01", "2026-01-26", "2026-04-14", "2026-05-01"]

def generate_synthetic_master_table(
    num_days: int = 180,
    start_date: str = "2026-01-01",
    seed: int = 42,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Generates synthetic master table CSV for DEMO MODE testing."""
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")
    
    rows = []
    
    for store in STORES:
        for product in PRODUCTS:
            # Initial inventory state
            current_stock = product["reorder_level"] * 3
            pending_order_days = 0
            pending_order_qty = 0
            
            for date in dates:
                d_str = date.strftime("%Y-%m-%d")
                day_of_week = date.dayofweek
                is_weekend = 1 if day_of_week in [5, 6] else 0
                is_festival = 1 if d_str in FESTIVAL_DATES else 0
                is_holiday = 1 if d_str in HOLIDAY_DATES else 0
                is_local_event = 1 if (np.random.rand() < 0.05) else 0

                # Weather simulation
                month = date.month
                temp_c = round(22 + 8 * np.sin(2 * np.pi * date.dayofyear / 365.0) + np.random.normal(0, 2), 1)
                rain_mm = round(max(0, np.random.exponential(scale=5) if month in [6, 7, 8] else np.random.exponential(scale=0.5)), 1)
                
                # Promotion simulation
                is_promo = 1 if (np.random.rand() < 0.15) else 0
                discount_pct = round(np.random.choice([0.0, 0.10, 0.15, 0.20]) if is_promo else 0.0, 2)
                selling_price = round(product["mrp"] * (1.0 - discount_pct), 2)
                
                # Unconstrained Demand calculation
                base = product["base_demand"] * store["multiplier"]
                weekend_mult = 1.35 if is_weekend else 1.0
                festival_mult = 1.6 if is_festival else 1.0
                promo_mult = (1.0 + discount_pct * 1.8) if is_promo else 1.0
                weather_mult = 1.2 if (product["category"] == "Beverages" and temp_c > 28) else 1.0
                noise = np.random.normal(1.0, 0.12)
                
                unconstrained_demand = max(0, int(round(base * weekend_mult * festival_mult * promo_mult * weather_mult * noise)))
                
                # Inventory Receiving
                received = 0
                if pending_order_days > 0:
                    pending_order_days -= 1
                    if pending_order_days == 0:
                        received = pending_order_qty
                        pending_order_qty = 0
                
                opening_stock = current_stock + received
                
                # Actual Sales (constrained by available opening stock)
                units_sold = min(opening_stock, unconstrained_demand)
                closing_stock = opening_stock - units_sold
                current_stock = closing_stock
                
                # Reorder Trigger Logic
                lead_days = product["lead_days"]
                reorder_level = product["reorder_level"]
                if closing_stock <= reorder_level and pending_order_days == 0:
                    pending_order_qty = int(base * 7) # order 7 days of supply
                    pending_order_days = lead_days
                
                row = {
                    "date": d_str,
                    "store_id": store["store_id"],
                    "product_id": product["product_id"],
                    "product_name": product["name"],
                    "units_sold": float(units_sold),
                    "selling_price": selling_price,
                    "discount_pct": discount_pct,
                    "promotion_flag": is_promo,
                    "opening_stock": float(opening_stock),
                    "received": float(received),
                    "closing_stock": float(closing_stock),
                    "reorder_level": float(reorder_level),
                    "lead_days": lead_days,
                    "category": product["category"],
                    "sub_category": product["sub_category"],
                    "brand": product["brand"],
                    "mrp": product["mrp"],
                    "cost_price": product["cost_price"],
                    "shelf_life_days": product["shelf_life_days"],
                    "store_type": store["store_type"],
                    "city": store["city"],
                    "floor_area_sqft": store["floor_area_sqft"],
                    "avg_daily_customers": store["avg_daily_customers"],
                    "temp_c": temp_c,
                    "rain_mm": rain_mm,
                    "holiday": is_holiday,
                    "festival": is_festival,
                    "weekend": is_weekend,
                    "local_event": is_local_event,
                    "is_synthetic": 1
                }
                rows.append(row)
                
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store_id", "product_id", "date"]).reset_index(drop=True)
    
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, index=False)
        logger.info(f"Synthetic master table saved to {out_p} ({len(df)} rows)")
        
    return df
