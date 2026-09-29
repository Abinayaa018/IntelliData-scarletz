# STOCKSENSE Feature Dictionary & Data Leakage Audit

This document describes every engineered feature in the STOCKSENSE platform, its business rationale, 
and formal proof of **Zero Data Leakage**.

| Feature Name | Group | Business Rationale | Leakage Safety Proof |
| :--- | :--- | :--- | :--- |
| `day_of_week` | Time | Day of week (0=Mon, 6=Sun) | Derived directly from calendar date t |
| `weekend_flag` | Time | 1 if Saturday/Sunday, else 0 | Derived directly from calendar date t |
| `month` | Time | Month of year (1..12) | Derived directly from calendar date t |
| `week_no` | Time | ISO week number of year | Derived directly from calendar date t |
| `day_of_month` | Time | Day of month (1..31) | Derived directly from calendar date t |
| `festival_flag` | Time | 1 if date is a major festival | Public calendar event known in advance |
| `holiday_flag` | Time | 1 if date is a public holiday | Public calendar event known in advance |
| `days_to_next_festival` | Time | Days remaining until next festival | Known calendar schedule |
| `days_since_last_festival` | Time | Days elapsed since prior festival | Known calendar schedule |
| `days_to_next_holiday` | Time | Days remaining until next public holiday | Known calendar schedule |
| `lag_1` | Lag | Units sold 1 day prior (t-1) | Strictly shifted by 1 day |
| `lag_7` | Lag | Units sold 7 days prior (t-7) | Strictly shifted by 7 days |
| `lag_14` | Lag | Units sold 14 days prior (t-14) | Strictly shifted by 14 days |
| `lag_28` | Lag | Units sold 28 days prior (t-28) | Strictly shifted by 28 days |
| `rolling_mean_7` | Rolling | 7-day moving average of demand [t-7 to t-1] | Shifted by 1 day prior to rolling mean |
| `rolling_std_7` | Rolling | 7-day demand standard deviation [t-7 to t-1] | Shifted by 1 day prior to rolling std |
| `rolling_max_7` | Rolling | 7-day peak demand [t-7 to t-1] | Shifted by 1 day prior to rolling max |
| `rolling_min_7` | Rolling | 7-day minimum demand [t-7 to t-1] | Shifted by 1 day prior to rolling min |
| `rolling_mean_14` | Rolling | 14-day moving average demand [t-14 to t-1] | Shifted by 1 day prior to rolling mean |
| `rolling_std_14` | Rolling | 14-day demand standard deviation [t-14 to t-1] | Shifted by 1 day prior to rolling std |
| `rolling_mean_28` | Rolling | 28-day moving average demand [t-28 to t-1] | Shifted by 1 day prior to rolling mean |
| `recent_growth_rate` | Rolling | 7-day rolling mean vs prior 7-day rolling mean | Shifted window comparison |
| `coefficient_of_variation_14` | Rolling | Demand volatility index (std_14 / mean_14) | Shifted rolling window statistics |
| `days_of_inventory` | Inventory | Stock on hand divided by rolling 7-day avg demand | Opening/Closing stock at t vs shifted avg demand |
| `inventory_to_demand_ratio` | Inventory | Stock on hand divided by 14-day avg demand | Opening/Closing stock at t vs shifted avg demand |
| `reorder_gap` | Inventory | Current stock minus store reorder level | Known inventory parameter at t |
| `lead_time_demand` | Inventory | Expected demand during supplier lead time | Rolling avg demand x lead days |
| `stock_cover_vs_lead_time` | Inventory | Ratio of days of inventory to lead days | Known inventory parameter at t |
| `past_stockout_freq_30` | Inventory | Stockout event frequency in prior 30 days | Shifted historical stock level series |
| `days_since_last_stockout` | Inventory | Days elapsed since last zero-stock event | Shifted historical stock level series |
| `discount_pct` | Price/Promo | Percentage price discount applied | Known price tag at prediction time t |
| `price_to_mrp_ratio` | Price/Promo | Selling price divided by MRP | Known price tag at prediction time t |
| `price_change` | Price/Promo | Price difference vs previous day | Shifted price change |
| `promotion_flag` | Price/Promo | 1 if active promotional campaign at t | Known marketing schedule at t |
| `promo_days_last_7` | Price/Promo | Active promo days in prior 7 days | Shifted marketing schedule |
| `days_since_last_promo` | Price/Promo | Days elapsed since prior promo | Shifted marketing schedule |
| `historical_promo_lift` | Price/Promo | Product promo lift multiplier computed on historical data | Computed on past training window |
| `perishability_flag` | Product/Store | 1 if product shelf life <= 7 days | Static product master attribute |
| `temp_c` | External | Ambient temperature in Celsius | Measured/forecasted daily weather |
| `rain_mm` | External | Daily rainfall in millimeters | Measured/forecasted daily weather |
| `temp_change` | External | Temperature change vs previous day | Shifted weather change |
| `local_event` | External | 1 if local city event occurs | Public local event schedule |
| `low_history_flag` | Quality | 1 if store/product has <14 days historical observations | Calculated observation count up to t |