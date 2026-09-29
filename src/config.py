"""
STOCKSENSE Configuration Management
Handles YAML config parsing, column renaming, dynamic schema mapping, and global path management.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging

logger = logging.getLogger("stocksense.config")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

class Config:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.raw_config = self._load_yaml()
        self.root_dir = Path(__file__).resolve().parent.parent
        
    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            logger.warning(f"Config file not found at {self.config_path}. Using fallback defaults.")
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def column_mapping(self) -> Dict[str, str]:
        return self.raw_config.get("column_mapping", {})

    @property
    def feature_params(self) -> Dict[str, Any]:
        return self.raw_config.get("features", {
            "sparse_history_min_days": 14,
            "perishability_shelf_life_days": 7,
            "lag_days": [1, 7, 14, 28],
            "rolling_windows": [7, 14, 28]
        })

    @property
    def target_params(self) -> Dict[str, Any]:
        return self.raw_config.get("targets", {
            "forecast_horizon_days": 7,
            "stockout_threshold_demand_multiplier": 1.0
        })

    @property
    def model_params(self) -> Dict[str, Any]:
        return self.raw_config.get("models", {
            "random_seed": 42,
            "time_series_splits": 5,
            "risk_thresholds": {"high": 0.70, "medium": 0.40},
            "service_level_z": 1.645
        })

    def apply_column_mapping(self, df):
        """Renames raw columns in dataframe to canonical STOCKSENSE internal column names."""
        mapping = self.column_mapping
        inv_mapping = {v: k for k, v in mapping.items() if k in df.columns}
        if inv_mapping:
            logger.info(f"Applying column mapping: {inv_mapping}")
            df = df.rename(columns=inv_mapping)
        return df

def load_config(config_path: Optional[str] = None) -> Config:
    p = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return Config(p)
