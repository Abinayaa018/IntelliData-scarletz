"""
Time Features Group
Computes calendar, weekend, holiday, and festival distance metrics.
LEAKAGE GUARD: All calendar metrics depend strictly on the calendar date 't'.
"""

import pandas as pd
import numpy as np

def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    dt = pd.to_datetime(df["date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["weekend_flag"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = dt.dt.month
    df["week_no"] = dt.dt.isocalendar().week.astype(int)
    df["day_of_month"] = dt.dt.day
    
    # Holiday / Festival flags (default to 0 if column missing)
    df["festival_flag"] = df["festival"].astype(int) if "festival" in df.columns else 0
    df["holiday_flag"] = df["holiday"].astype(int) if "holiday" in df.columns else 0

    # Calculate distance to next / since last festival/holiday across the dataset
    dates = pd.Series(df["date"].unique()).sort_values().reset_index(drop=True)
    
    fest_dates = df[df["festival_flag"] == 1]["date"].unique()
    hol_dates = df[df["holiday_flag"] == 1]["date"].unique()

    days_to_next_fest = []
    days_since_last_fest = []
    days_to_next_hol = []

    for d in df["date"]:
        # Next festival
        future_fests = [f for f in fest_dates if f >= d]
        days_to_next_fest.append((future_fests[0] - d).days if future_fests else 999)
        
        # Past festival
        past_fests = [f for f in fest_dates if f <= d]
        days_since_last_fest.append((d - past_fests[-1]).days if past_fests else 999)

        # Next holiday
        future_hols = [h for h in hol_dates if h >= d]
        days_to_next_hol.append((future_hols[0] - d).days if future_hols else 999)

    df["days_to_next_festival"] = days_to_next_fest
    df["days_since_last_festival"] = days_since_last_fest
    df["days_to_next_holiday"] = days_to_next_hol

    return df
