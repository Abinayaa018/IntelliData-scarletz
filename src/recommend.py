"""
STOCKSENSE Recommendation Engine & Decision Intelligence Module
Implements safety stock calculations, replenishment orders, financial lost sales estimation,
overstock/expiry risk alerts, rule-based manager actions, and a real-time What-If simulator.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("stocksense.recommend")

class RecommendationEngine:
    def __init__(
        self, 
        service_level_z: float = 1.645, 
        high_risk_thresh: float = 0.70, 
        med_risk_thresh: float = 0.40,
        pack_size: int = 1
    ):
        self.z = service_level_z
        self.high_risk_thresh = high_risk_thresh
        self.med_risk_thresh = med_risk_thresh
        self.pack_size = pack_size

    def calculate_recommendations(
        self, 
        df: pd.DataFrame, 
        forecast_col: str = "pred_7_day_demand", 
        prob_col: str = "pred_stockout_prob",
        std_error_per_cat: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Generates actionable replenishment recommendations for every Store x Product row.
        Outputs exact required schema: Store, Product, Current Stock, 7-Day Forecast, 
        Stock-out Probability, Risk, Recommended Order, Key Reasons, Manager Action, Estimated Lost Sales.
        """
        df = df.copy()
        
        # Default forecast error std per category if not provided
        default_cat_std = {"Dairy": 8.0, "Bakery": 6.0, "Beverages": 10.0, "Pantry": 5.0, "Personal Care": 3.0}
        cat_std_map = std_error_per_cat or default_cat_std

        recommendations = []

        for idx, row in df.iterrows():
            store = str(row.get("store_id", "S01"))
            prd_id = str(row.get("product_id", "P001"))
            prd_name = str(row.get("product_name", f"Product {prd_id}"))
            category = str(row.get("category", "General"))
            
            curr_stock = float(row.get("closing_stock", row.get("opening_stock", 0.0)))
            incoming_stock = float(row.get("received", 0.0))
            forecast_7d = max(0.0, float(row.get(forecast_col, 0.0)))
            prob_stockout = float(np.clip(row.get(prob_col, 0.0), 0.0, 1.0))
            price = float(row.get("selling_price", row.get("mrp", 10.0)))
            lead_days = float(row.get("lead_days", 3))
            shelf_life = float(row.get("shelf_life_days", 30))
            low_hist = int(row.get("low_history_flag", 0))

            # 1. Safety Stock = z * error_std * sqrt(lead_days)
            error_std = cat_std_map.get(category, 7.0)
            safety_stock = self.z * error_std * np.sqrt(lead_days)
            
            # 2. Recommended Stock & Reorder Quantity
            recommended_stock = forecast_7d + safety_stock
            raw_reorder = max(0.0, recommended_stock - curr_stock - incoming_stock)
            
            # Round to pack size multiple
            reorder_qty = int(np.ceil(raw_reorder / self.pack_size) * self.pack_size) if raw_reorder > 0 else 0

            # 3. Risk Level
            if prob_stockout >= self.high_risk_thresh:
                risk_level = "HIGH"
            elif prob_stockout >= self.med_risk_thresh:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            # 4. Estimated Lost Sales (unfulfilled demand * selling price)
            unfulfilled_demand = max(0.0, forecast_7d - (curr_stock + incoming_stock))
            estimated_lost_sales = round(unfulfilled_demand * price, 2)

            # 5. Overstock & Expiry Risk Assessment
            days_of_inv = curr_stock / ((forecast_7d / 7.0) + 1e-5)
            is_overstocked = days_of_inv > 21.0
            is_expiry_risk = (shelf_life <= 7) and (curr_stock > forecast_7d)

            # 6. Manager Action text generator rules
            if is_expiry_risk:
                action = "Reduce order: severe perishable expiry risk"
            elif is_overstocked:
                action = "Reduce order / Hold: overstock alert (>21 days cover)"
            elif risk_level == "HIGH" and reorder_qty > 0:
                action = f"Raise replenishment order IMMEDIATELY (+{reorder_qty} units)"
            elif risk_level == "MEDIUM" and reorder_qty > 0:
                action = f"Order within 2 days (+{reorder_qty} units)"
            else:
                action = "Monitor stock levels — adequate inventory cover"

            if low_hist == 1:
                action += " [Note: Sparse history - manual review recommended]"

            # 7. Key Reasons text
            reasons_list = []
            if prob_stockout >= 0.7:
                reasons_list.append("High stockout probability")
            if unfulfilled_demand > 0:
                reasons_list.append(f"Unfulfilled demand ({unfulfilled_demand:.0f} units)")
            if curr_stock < row.get("reorder_level", 0):
                reasons_list.append("Stock below reorder level")
            if not reasons_list:
                reasons_list.append("Stable demand velocity")

            reasons_text = ", ".join(reasons_list)

            rec_row = {
                "Store": store,
                "Product": prd_name,
                "Current Stock": int(round(curr_stock)),
                "7-Day Forecast": int(round(forecast_7d)),
                "Stock-out Probability": round(prob_stockout, 3),
                "Risk": risk_level,
                "Recommended Order": reorder_qty,
                "Key Reasons": reasons_text,
                "Manager Action": action,
                "Estimated Lost Sales": estimated_lost_sales
            }
            recommendations.append(rec_row)

        return pd.DataFrame(recommendations)

class WhatIfSimulator:
    """Simulates changes in discounts, supplier delays, and festival demand spikes in real-time."""
    def __init__(self, reg_model, clf_model, engine: RecommendationEngine):
        self.reg_model = reg_model
        self.clf_model = clf_model
        self.engine = engine

    def simulate(
        self, 
        base_row_df: pd.DataFrame, 
        discount_pct_delta: float = 0.0, 
        extra_lead_days: int = 0,
        festival_uplift: bool = False
    ) -> Dict[str, Any]:
        """Runs what-if simulation for a single row dataframe."""
        sim_df = base_row_df.copy()
        
        # Apply parameter modifications
        if "discount_pct" in sim_df.columns:
            sim_df["discount_pct"] = np.clip(sim_df["discount_pct"].iloc[0] + discount_pct_delta, 0.0, 0.8)
            sim_df["promotion_flag"] = 1 if sim_df["discount_pct"].iloc[0] > 0 else 0
            
        if "lead_days" in sim_df.columns:
            sim_df["lead_days"] = sim_df["lead_days"].iloc[0] + extra_lead_days

        if festival_uplift:
            sim_df["festival_flag"] = 1
            sim_df["days_to_next_festival"] = 0

        # Predict new forecast & probability
        new_forecast = max(0.0, float(self.reg_model.predict(sim_df)[0]))
        new_prob = float(np.clip(self.clf_model.predict_proba(sim_df)[0, 1], 0.0, 1.0))
        
        sim_df["pred_7_day_demand"] = new_forecast
        sim_df["pred_stockout_prob"] = new_prob

        rec_table = self.engine.calculate_recommendations(sim_df)
        rec_info = rec_table.iloc[0].to_dict()

        return {
            "new_forecast": round(new_forecast, 1),
            "new_prob": round(new_prob, 3),
            "new_risk": rec_info["Risk"],
            "recommended_order": rec_info["Recommended Order"],
            "estimated_lost_sales": rec_info["Estimated Lost Sales"],
            "manager_action": rec_info["Manager Action"]
        }
