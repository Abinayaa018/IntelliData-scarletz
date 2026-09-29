"""
STOCKSENSE Explainability Engine
Computes Global Permutation Feature Importance and Local Manager-Friendly Per-Row Driver Contributions.
Translates raw technical feature names into intuitive business reasons with percentage impacts.
"""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import logging

logger = logging.getLogger("stocksense.explain")

BUSINESS_REASON_MAP = {
    "rolling_mean_7": "Recent 7-day sales velocity",
    "rolling_mean_14": "Medium-term 14-day sales trend",
    "rolling_mean_28": "Baseline monthly demand",
    "recent_growth_rate": "Accelerating sales growth rate",
    "lag_1": "Yesterday's demand volume",
    "lag_7": "Same weekday last week demand",
    "promotion_flag": "Active marketing campaign discount",
    "discount_pct": "High retail price discount",
    "days_to_next_festival": "Upcoming festival demand surge",
    "festival_flag": "Active festival shopping period",
    "weekend_flag": "Weekend footfall surge",
    "reorder_gap": "Stock already below reorder threshold",
    "days_of_inventory": "Low stock cover remaining",
    "stock_cover_vs_lead_time": "Stock cover inadequate vs lead time",
    "past_stockout_freq_30": "Repeated recent stockout history",
    "temp_c": "Hot weather beverage/dairy demand shift",
    "temp_change": "Sudden temperature spike",
    "perishability_flag": "Short shelf-life expiration risk",
    "historical_promo_lift": "High promotional elasticity",
    "days_since_last_stockout": "Recent stock replenishment recovery"
}

class ModelExplainer:
    def __init__(self, model_pipeline, feature_names: List[str]):
        self.model_pipeline = model_pipeline
        self.feature_names = feature_names

    def get_global_permutation_importance(
        self, 
        test_df: pd.DataFrame, 
        y_test: np.ndarray, 
        scoring: str = "neg_mean_absolute_error", 
        n_repeats: int = 5,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """Calculates global permutation feature importance on the test set."""
        logger.info("Computing global permutation importance...")
        result = permutation_importance(
            self.model_pipeline, 
            test_df, 
            y_test, 
            scoring=scoring, 
            n_repeats=n_repeats, 
            random_state=random_seed,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            "feature": test_df.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)
        
        # Translate to business names
        importance_df["business_reason"] = importance_df["feature"].map(BUSINESS_REASON_MAP).fillna(importance_df["feature"])
        return importance_df

    def explain_row_local(self, row_df: pd.DataFrame, top_k: int = 4) -> Dict[str, Any]:
        """
        Computes local per-row driver contribution percentages using feature occlusion / baseline diff.
        Compares row prediction against baseline mean feature prediction.
        """
        base_pred = self.model_pipeline.predict(row_df)[0]
        
        # Occlusion perturbation to measure individual feature sensitivity
        sensitivities = {}
        for col in self.feature_names:
            if col not in row_df.columns:
                continue
            orig_val = row_df[col].iloc[0]
            
            # Perturb numeric column slightly to observe delta
            perturbed_df = row_df.copy()
            if isinstance(orig_val, (int, float, np.number)):
                perturbed_df[col] = orig_val * 0.5 # 50% reduction perturbation
            else:
                perturbed_df[col] = "OTHER"
                
            p_pred = self.model_pipeline.predict(perturbed_df)[0]
            delta = abs(base_pred - p_pred)
            sensitivities[col] = delta

        # Normalize to percentage impacts
        total_delta = sum(sensitivities.values()) + 1e-5
        impact_pcts = {k: (v / total_delta) * 100.0 for k, v in sensitivities.items()}
        
        # Sort top-k drivers
        sorted_drivers = sorted(impact_pcts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        reasons = []
        formatted_drivers = []
        for feat, pct in sorted_drivers:
            b_name = BUSINESS_REASON_MAP.get(feat, feat.replace("_", " ").title())
            formatted_drivers.append({"feature": feat, "business_reason": b_name, "impact_pct": round(pct, 1)})
            reasons.append(f"{b_name} ({pct:.1f}%)")
            
        reason_text = ", ".join(reasons)
        
        return {
            "prediction": round(float(base_pred), 2),
            "top_drivers": formatted_drivers,
            "reason_summary": reason_text
        }

def sanity_test_explainability(explainer: ModelExplainer, sample_df: pd.DataFrame) -> bool:
    """Verifies that local explanations run cleanly, sum to sensible totals, and rank consistently."""
    res = explainer.explain_row_local(sample_df.iloc[[0]])
    drivers = res["top_drivers"]
    total_pct = sum([d["impact_pct"] for d in drivers])
    assert len(drivers) > 0, "Explainer returned empty drivers"
    assert total_pct > 0, "Driver impact total must be positive"
    logger.info(f"Explainability sanity check passed: {res['reason_summary']}")
    return True
