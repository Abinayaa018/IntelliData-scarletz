# STOCKSENSE Column Mapping & Data Handling Specification

This report documents the mapping from the 5 Kaggle raw dataset tables (`transactions.csv`, `inventory.csv`, `products.csv`, `stores.csv`, `external_factors.csv`) to the canonical STOCKSENSE master table schema (`data/processed/master_table.csv`).

---

## 1. Schema Mapping Table

| Kaggle Raw Field | Source Table | Master Table Column | Transformation / Rule | Confidence |
| :--- | :--- | :--- | :--- | :--- |
| `date` | `transactions.csv` | `date` | Parsed to datetime (`YYYY-MM-DD`) | 100% (High) |
| `store_id` | `transactions.csv` | `store_id` | String cast, whitespace stripped | 100% (High) |
| `product_id` | `transactions.csv` | `product_id` | String cast, whitespace stripped | 100% (High) |
| `sum(quantity)` | `transactions.csv` | `units_sold` | Daily sum of non-negative transaction quantities per grain | 100% (High) |
| `mean(selling_price)` | `transactions.csv` | `selling_price` | Daily mean selling price per grain | 100% (High) |
| `mean(discount_pct)` | `transactions.csv` | `discount_pct` | Daily mean discount percentage per grain | 100% (High) |
| `max(promotion_flag)` | `transactions.csv` | `promotion_flag` | 1 if any transaction had active promo, else 0 | 100% (High) |
| `opening` | `inventory.csv` | `opening_stock` | Direct join on `date x store_id x product_id` | 100% (High) |
| `received` | `inventory.csv` | `received` | Direct join on `date x store_id x product_id` | 100% (High) |
| `closing` | `inventory.csv` | `closing_stock` | Direct join on `date x store_id x product_id` | 100% (High) |
| `reorder_lvl` | `inventory.csv` | `reorder_level` | Direct join on `date x store_id x product_id` | 100% (High) |
| `lead_days` | `inventory.csv` | `lead_days` | Direct join on `date x store_id x product_id` | 100% (High) |
| `category` | `products.csv` | `category` | Standardized title case text (`Dairy`, `Bakery`, `Beverages`, `Pantry`, `Personal Care`) | 100% (High) |
| `sub_category` | `products.csv` | `sub_category` | Title case text standardization | 100% (High) |
| `brand` | `products.csv` | `brand` | Direct join on `product_id` | 100% (High) |
| `mrp` | `products.csv` | `mrp` | Direct join on `product_id` | 100% (High) |
| `cost_price` | `products.csv` | `cost_price` | Direct join on `product_id` | 100% (High) |
| `shelf_life_days` | `products.csv` | `shelf_life_days` | Direct join on `product_id` | 100% (High) |
| `store_type` | `stores.csv` | `store_type` | Direct join on `store_id` | 100% (High) |
| `city` | `stores.csv` | `city` | Direct join on `store_id` | 100% (High) |
| `floor_area_sqft` | `stores.csv` | `floor_area_sqft` | Direct join on `store_id` | 100% (High) |
| `avg_daily_customers` | `stores.csv` | `avg_daily_customers` | Direct join on `store_id` | 100% (High) |
| `temp_c` | `external_factors.csv` | `temp_c` | Join on `date x city`. Impute 18 missing values via city forward-fill | 100% (High) |
| `rain_mm` | `external_factors.csv` | `rain_mm` | Join on `date x city` | 100% (High) |
| `holiday` | `external_factors.csv` | `holiday` | Join on `date x city` | 100% (High) |
| `festival` | `external_factors.csv` | `festival` | Join on `date x city` | 100% (High) |
| `weekend` | `external_factors.csv` | `weekend` | Join on `date x city` | 100% (High) |
| `local_event` | `external_factors.csv` | `local_event` | Join on `date x city` | 100% (High) |

---

## 2. Handling Rules for Missing/Derived Fields

1. **Inventory Fields**: All inventory fields (`opening`, `received`, `sold`, `closing`, `reorder_lvl`, `lead_days`) are directly available in `inventory.csv`. No synthetic inventory generation is needed. An inventory arithmetic check (`closing == opening + received - units_sold`) will flag any discrepancies as data quality flags in `data/processed/data_quality_flags.csv`.
2. **External Factors**: Temperature (`temp_c`) contains 18 missing values (1.85%). These are forward-filled per city, with city-wide median as fallback. Calendar variables (`holiday`, `festival`, `weekend`, `local_event`) are 100% populated in `external_factors.csv`.
3. **Product & Store Attributes**: Category casing variants (e.g. `Beverages`, `beverage`, `BEVERAGES`) are standardized to Title Case via config dictionary mapping. No product or store attributes are missing.
