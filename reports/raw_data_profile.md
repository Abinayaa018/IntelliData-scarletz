# STOCKSENSE Raw Data Profiling Report

## File: `external_factors.csv`
- **Shape**: 972 rows, 8 columns
- **Total Duplicates**: 0 rows
- **Date Range (`date`)**: 2026-01-01 to 2026-08-31 (243 unique dates)

### Columns & Data Types
| Column      | Dtype   |   Missing Count | Missing %   |   Unique Count | Stats / Sample                                         |
|:------------|:--------|----------------:|:------------|---------------:|:-------------------------------------------------------|
| date        | object  |               0 | 0.00%       |            243 | Top values: ['2026-01-01', '2026-01-02', '2026-01-03'] |
| city        | object  |               0 | 0.00%       |              4 | Top values: ['Coimbatore', 'Chennai', 'Madurai']       |
| temp_c      | float64 |              18 | 1.85%       |            103 | Min: 24.6, Max: 36.0, Mean: 30.66                      |
| rain_mm     | float64 |               0 | 0.00%       |            106 | Min: 0.0, Max: 25.0, Mean: 1.38                        |
| holiday     | int64   |               0 | 0.00%       |              2 | Min: 0.0, Max: 1.0, Mean: 0.02                         |
| festival    | int64   |               0 | 0.00%       |              2 | Min: 0.0, Max: 1.0, Mean: 0.02                         |
| weekend     | int64   |               0 | 0.00%       |              2 | Min: 0.0, Max: 1.0, Mean: 0.29                         |
| local_event | int64   |               0 | 0.00%       |              2 | Min: 0.0, Max: 1.0, Mean: 0.04                         |

### Sample Rows (Head 3)
| date       | city       |   temp_c |   rain_mm |   holiday |   festival |   weekend |   local_event |
|:-----------|:-----------|---------:|----------:|----------:|-----------:|----------:|--------------:|
| 2026-01-01 | Coimbatore |     27.2 |       4   |         1 |          0 |         0 |             0 |
| 2026-01-01 | Chennai    |     33.4 |       7.4 |         1 |          0 |         0 |             0 |
| 2026-01-01 | Madurai    |     29.8 |       1.2 |         1 |          0 |         0 |             0 |

---

## File: `inventory.csv`
- **Shape**: 29160 rows, 9 columns
- **Total Duplicates**: 0 rows
- **Date Range (`date`)**: 2026-01-01 to 2026-08-31 (243 unique dates)

### Columns & Data Types
| Column      | Dtype   |   Missing Count | Missing %   |   Unique Count | Stats / Sample                                         |
|:------------|:--------|----------------:|:------------|---------------:|:-------------------------------------------------------|
| date        | object  |               0 | 0.00%       |            243 | Top values: ['2026-01-01', '2026-01-02', '2026-01-03'] |
| store_id    | object  |               0 | 0.00%       |              4 | Top values: ['S01', 'S02', 'S03']                      |
| product_id  | object  |               0 | 0.00%       |             30 | Top values: ['P101', 'P102', 'P103']                   |
| opening     | int64   |               0 | 0.00%       |            864 | Min: 11.0, Max: 1127.0, Mean: 195.0                    |
| received    | int64   |               0 | 0.00%       |            461 | Min: 0.0, Max: 537.0, Mean: 36.75                      |
| sold        | int64   |               0 | 0.00%       |            164 | Min: 3.0, Max: 185.0, Mean: 36.42                      |
| closing     | int64   |               0 | 0.00%       |            864 | Min: 11.0, Max: 1127.0, Mean: 195.33                   |
| reorder_lvl | int64   |               0 | 0.00%       |            321 | Min: 15.0, Max: 861.0, Mean: 153.96                    |
| lead_days   | int64   |               0 | 0.00%       |              6 | Min: 1.0, Max: 6.0, Mean: 2.55                         |

### Sample Rows (Head 3)
| date       | store_id   | product_id   |   opening |   received |   sold |   closing |   reorder_lvl |   lead_days |
|:-----------|:-----------|:-------------|----------:|-----------:|-------:|----------:|--------------:|------------:|
| 2026-01-01 | S01        | P101         |        82 |         82 |     24 |       140 |            87 |           2 |
| 2026-01-01 | S01        | P102         |       293 |        167 |     59 |       401 |           391 |           4 |
| 2026-01-01 | S01        | P103         |       124 |        282 |     56 |       350 |           290 |           3 |

---

## File: `products.csv`
- **Shape**: 30 rows, 8 columns
- **Total Duplicates**: 0 rows

### Columns & Data Types
| Column          | Dtype   |   Missing Count | Missing %   |   Unique Count | Stats / Sample                                      |
|:----------------|:--------|----------------:|:------------|---------------:|:----------------------------------------------------|
| product_id      | object  |               0 | 0.00%       |             30 | Top values: ['P101', 'P102', 'P103']                |
| category        | object  |               0 | 0.00%       |              8 | Top values: ['Dairy', 'Personal Care', 'Household'] |
| sub_category    | object  |               0 | 0.00%       |             30 | Top values: ['Milk', 'Curd', 'Butter']              |
| brand           | object  |               0 | 0.00%       |             20 | Top values: ['Aavin', 'CleanMax', 'FrostFarm']      |
| mrp             | int64   |               0 | 0.00%       |             21 | Min: 25.0, Max: 220.0, Mean: 104.33                 |
| cost_price      | int64   |               0 | 0.00%       |             28 | Min: 16.0, Max: 165.0, Mean: 75.67                  |
| shelf_life_days | int64   |               0 | 0.00%       |             12 | Min: 3.0, Max: 900.0, Mean: 352.0                   |
| supplier_id     | object  |               0 | 0.00%       |             14 | Top values: ['SUP01', 'SUP08', 'SUP12']             |

### Sample Rows (Head 3)
| product_id   | category   | sub_category   | brand     |   mrp |   cost_price |   shelf_life_days | supplier_id   |
|:-------------|:-----------|:---------------|:----------|------:|-------------:|------------------:|:--------------|
| P101         | Dairy      | Milk           | Aavin     |    60 |           52 |                 3 | SUP01         |
| P102         | Dairy      | Curd           | Aavin     |    45 |           38 |                 7 | SUP01         |
| P103         | Dairy      | Butter         | MilkyMist |   110 |           92 |                60 | SUP02         |

---

## File: `stores.csv`
- **Shape**: 4 rows, 6 columns
- **Total Duplicates**: 0 rows

### Columns & Data Types
| Column              | Dtype   |   Missing Count | Missing %   |   Unique Count | Stats / Sample                                        |
|:--------------------|:--------|----------------:|:------------|---------------:|:------------------------------------------------------|
| store_id            | object  |               0 | 0.00%       |              4 | Top values: ['S01', 'S02', 'S03']                     |
| city                | object  |               0 | 0.00%       |              4 | Top values: ['Coimbatore', 'Chennai', 'Madurai']      |
| store_type          | object  |               0 | 0.00%       |              3 | Top values: ['Supermarket', 'Hypermarket', 'Express'] |
| floor_area_sqft     | int64   |               0 | 0.00%       |              4 | Min: 4200.0, Max: 18000.0, Mean: 9575.0               |
| avg_daily_customers | int64   |               0 | 0.00%       |              4 | Min: 720.0, Max: 2850.0, Mean: 1475.0                 |
| region              | object  |               0 | 0.00%       |              4 | Top values: ['West', 'North', 'South']                |

### Sample Rows (Head 3)
| store_id   | city       | store_type   |   floor_area_sqft |   avg_daily_customers | region   |
|:-----------|:-----------|:-------------|------------------:|----------------------:|:---------|
| S01        | Coimbatore | Supermarket  |              8500 |                  1250 | West     |
| S02        | Chennai    | Hypermarket  |             18000 |                  2850 | North    |
| S03        | Madurai    | Express      |              4200 |                   720 | South    |

---

## File: `transactions.csv`
- **Shape**: 67344 rows, 11 columns
- **Total Duplicates**: 1 rows
- **Date Range (`date`)**: 2026-01-01 to 2026-08-31 (243 unique dates)

### Columns & Data Types
| Column         | Dtype   |   Missing Count | Missing %   |   Unique Count | Stats / Sample                                         |
|:---------------|:--------|----------------:|:------------|---------------:|:-------------------------------------------------------|
| transaction_id | object  |               0 | 0.00%       |          67343 | Top values: ['T100011', 'T100012', 'T100013']          |
| date           | object  |               0 | 0.00%       |            243 | Top values: ['2026-04-06', '2026-06-18', '2026-06-11'] |
| store_id       | object  |               0 | 0.00%       |              4 | Top values: ['S02', 'S03', 'S01']                      |
| product_id     | object  |               0 | 0.00%       |             30 | Top values: ['P403', 'P201', 'P505']                   |
| quantity       | int64   |               0 | 0.00%       |            156 | Min: -4.0, Max: 172.0, Mean: 15.77                     |
| selling_price  | float64 |               0 | 0.00%       |           9964 | Min: 19.41, Max: 225.63, Mean: 103.52                  |
| discount_pct   | float64 |               0 | 0.00%       |              5 | Min: 0.0, Max: 20.0, Mean: 0.59                        |
| promotion_flag | int64   |               0 | 0.00%       |              2 | Min: 0.0, Max: 1.0, Mean: 0.12                         |
| customer_id    | object  |               0 | 0.00%       |           4000 | Top values: ['C2814', 'C1442', 'C2861']                |
| payment_mode   | object  |               0 | 0.00%       |              3 | Top values: ['UPI', 'Card', 'Cash']                    |
| hour           | int64   |               0 | 0.00%       |             14 | Min: 8.0, Max: 21.0, Mean: 15.31                       |

### Sample Rows (Head 3)
| transaction_id   | date       | store_id   | product_id   |   quantity |   selling_price |   discount_pct |   promotion_flag | customer_id   | payment_mode   |   hour |
|:-----------------|:-----------|:-----------|:-------------|-----------:|----------------:|---------------:|-----------------:|:--------------|:---------------|-------:|
| T100001          | 2026-01-01 | S01        | P101         |          3 |           59.95 |              0 |                0 | C1725         | UPI            |     18 |
| T100002          | 2026-01-01 | S01        | P101         |         21 |           60.2  |              0 |                0 | C2216         | UPI            |     19 |
| T100003          | 2026-01-01 | S01        | P102         |         42 |           44.74 |              0 |                0 | C3971         | UPI            |     16 |

---
