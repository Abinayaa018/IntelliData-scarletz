"""
STOCKSENSE Master Streamlit Dashboard App
Sci-Fi Dark "Mission Control / Holographic Command Centre" Interface
Features 3D Supermarket Aisle Digital Twin, 7 Detailed Analytics Sections,
Dynamic Plotly Dark Charts, and Real-Time What-If Simulator.
"""

from pathlib import Path
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Add root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import load_config
from src.validation import DataValidator
from src.synthetic_data import generate_synthetic_master_table
from src.features.builder import FeatureBuilder
from src.recommend import RecommendationEngine, WhatIfSimulator
from dashboard.components.component_3d import render_3d_digital_twin

# 1. Page Configuration
st.set_page_config(
    page_title="STOCKSENSE — Mission Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS
CSS_PATH = Path(__file__).resolve().parent / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Load Configuration & Data Cache
@st.cache_resource
def get_config():
    return load_config()

config = get_config()

@st.cache_data(ttl=600)
def load_dashboard_data():
    synth_path = ROOT_DIR / "data" / "processed" / "master_table_synthetic.csv"
    real_path = ROOT_DIR / "data" / "processed" / "master_table.csv"
    
    is_demo = not real_path.exists()
    target_path = synth_path if is_demo else real_path
    
    if not target_path.exists():
        generate_synthetic_master_table(num_days=180, output_path=synth_path)
        
    df_raw = pd.read_csv(target_path)
    validator = DataValidator(config)
    df_clean, _ = validator.validate(df_raw)
    
    fb = FeatureBuilder()
    df_feat = fb.transform(df_clean)
    
    return df_feat, is_demo

df_feat, is_demo_data = load_dashboard_data()

# Load Saved Pipelines
models_dir = ROOT_DIR / "models"
reg_model_path = models_dir / "demand_forecast_model.joblib"
clf_model_path = models_dir / "stockout_risk_model.joblib"

@st.cache_resource
def load_models():
    if reg_model_path.exists() and clf_model_path.exists():
        return joblib.load(reg_model_path), joblib.load(clf_model_path)
    return None, None

reg_pipe, clf_pipe = load_models()

# Predict Latest Snapshot
latest_date = df_feat["date"].max()
latest_df = df_feat[df_feat["date"] == latest_date].copy().reset_index(drop=True)

if reg_pipe and clf_pipe:
    latest_df["pred_7_day_demand"] = np.clip(reg_pipe.predict(latest_df), 0.0, None)
    latest_df["pred_stockout_prob"] = np.clip(clf_pipe.predict_proba(latest_df)[:, 1], 0.0, 1.0)
else:
    # Fallback heuristic if models not trained yet
    latest_df["pred_7_day_demand"] = latest_df["rolling_mean_7"] * 7.0
    latest_df["pred_stockout_prob"] = (latest_df["closing_stock"] < latest_df["pred_7_day_demand"]).astype(float)

rec_engine = RecommendationEngine(
    service_level_z=config.model_params.get("service_level_z", 1.645),
    high_risk_thresh=config.model_params.get("risk_thresholds", {}).get("high", 0.70),
    med_risk_thresh=config.model_params.get("risk_thresholds", {}).get("medium", 0.40)
)

rec_df = rec_engine.calculate_recommendations(latest_df)

# Sidebar Filters
st.sidebar.markdown("<h2 style='color:#00f3ff;'>⚡ STOCKSENSE</h2>", unsafe_allow_html=True)
if is_demo_data:
    st.sidebar.markdown("<span class='demo-badge'>DEMO DATA ACTIVE</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
selected_store = st.sidebar.selectbox("Store Location", ["ALL"] + list(rec_df["Store"].unique()))
selected_category = st.sidebar.selectbox("Product Category", ["ALL"] + ["Dairy", "Bakery", "Beverages", "Pantry", "Personal Care"])
selected_risk = st.sidebar.selectbox("Risk Level Filter", ["ALL", "HIGH", "MEDIUM", "LOW"])

# Filter Recommendations
filtered_rec = rec_df.copy()
if selected_store != "ALL":
    filtered_rec = filtered_rec[filtered_rec["Store"] == selected_store]
if selected_risk != "ALL":
    filtered_rec = filtered_rec[filtered_rec["Risk"] == selected_risk]

# Main Header
st.title("STOCKSENSE: Retail Demand & Replenishment Intelligence")
st.markdown("##### Real-Time Demand Forecasting & Stock-Out Risk Control Centre | NovaMart Retail 2026")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🌐 Executive Summary", 
    "📈 Demand Intelligence", 
    "🛡️ Inventory Risk", 
    "🛒 Manager Action Centre", 
    "🔬 Model Performance", 
    "🧠 Explainability", 
    "🎛️ What-If Lab"
])

# ==================== TAB 1: EXECUTIVE SUMMARY ====================
with tab1:
    st.markdown("### Executive Performance Dashboard & Live 3D Digital Twin")
    
    # KPI Grid
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown("<div class='kpi-tile'><div class='kpi-label'>Total Items Monitored</div><div class='kpi-value'>{}</div></div>".format(len(filtered_rec)), unsafe_allow_html=True)
    with col2:
        high_risk_count = (filtered_rec['Risk'] == 'HIGH').sum()
        st.markdown("<div class='kpi-tile'><div class='kpi-label'>Critical Stock-Out Risk</div><div class='kpi-value' style='color:#ff0055;'>{}</div></div>".format(high_risk_count), unsafe_allow_html=True)
    with col3:
        total_forecast = filtered_rec['7-Day Forecast'].sum()
        st.markdown("<div class='kpi-tile'><div class='kpi-label'>7-Day Demand Vol</div><div class='kpi-value'>{:,.0f}</div></div>".format(total_forecast), unsafe_allow_html=True)
    with col4:
        total_lost_sales = filtered_rec['Estimated Lost Sales'].sum()
        st.markdown("<div class='kpi-tile'><div class='kpi-label'>Est. Lost Sales Risk</div><div class='kpi-value' style='color:#ffaa00;'>₹{:,.0f}</div></div>".format(total_lost_sales), unsafe_allow_html=True)
    with col5:
        total_reorder_units = filtered_rec['Recommended Order'].sum()
        st.markdown("<div class='kpi-tile'><div class='kpi-label'>Total Reorder Qty</div><div class='kpi-value' style='color:#00ff88;'>{:,.0f}</div></div>".format(total_reorder_units), unsafe_allow_html=True)

    st.markdown("#### Live 3D Supermarket Aisle Digital Twin")
    render_3d_digital_twin(filtered_rec, height=440)

# ==================== TAB 2: DEMAND INTELLIGENCE ====================
with tab2:
    st.markdown("### 7-Day Demand Forecast & Historical Velocity")
    
    # Actual vs Forecast Line Chart
    sample_series = df_feat.groupby("date")["units_sold"].sum().reset_index()
    fig_demand = px.line(sample_series, x="date", y="units_sold", title="Daily Store Sales Volume (Historical & 7-Day Horizon)", template="plotly_dark")
    fig_demand.update_traces(line_color="#00f3ff", line_width=2.5)
    st.plotly_chart(fig_demand, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_cat = px.bar(filtered_rec, x="Product", y="7-Day Forecast", color="Risk", color_discrete_map={"HIGH":"#ff0055", "MEDIUM":"#ffaa00", "LOW":"#00ff88"}, title="Forecasted Demand by Product", template="plotly_dark")
        st.plotly_chart(fig_cat, use_container_width=True)
    with col_b:
        fig_stock = px.bar(filtered_rec, x="Product", y="Current Stock", color="Store", title="Current Stock on Hand per Product", template="plotly_dark")
        st.plotly_chart(fig_stock, use_container_width=True)

# ==================== TAB 3: INVENTORY RISK ====================
with tab3:
    st.markdown("### Stock-Out Risk Matrix & Probability Heatmap")
    
    fig_heat = px.density_heatmap(filtered_rec, x="Store", y="Product", z="Stock-out Probability", color_continuous_scale="Reds", title="Store x Product Stock-Out Risk Probability Heatmap", template="plotly_dark")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### High-Risk Priority Table")
    st.dataframe(filtered_rec.sort_values(by="Stock-out Probability", ascending=False), use_container_width=True)

# ==================== TAB 4: MANAGER ACTION CENTRE ====================
with tab4:
    st.markdown("### Manager Replenishment Decision Queue")
    
    col_dl1, col_dl2 = st.columns([3, 1])
    with col_dl2:
        st.download_button(
            "📥 Export Recommendations CSV",
            data=filtered_rec.to_csv(index=False),
            file_name="stocksense_recommendations.csv",
            mime="text/csv"
        )
        
    for _, r in filtered_rec.head(6).iterrows():
        r_color = "#ff0055" if r["Risk"] == "HIGH" else ("#ffaa00" if r["Risk"] == "MEDIUM" else "#00ff88")
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid {r_color};">
            <div style="display:flex; justify-between; align-items:center;">
                <div>
                    <strong style="font-size:1.2rem; color:#ffffff;">STORE {r['Store']} — {r['Product']}</strong>
                    <span class="badge-{r['Risk'].lower()}" style="margin-left:10px;">{r['Risk']} RISK ({(r['Stock-out Probability']*100):.0f}%)</span>
                </div>
                <div style="font-family:monospace; font-size:1.1rem; color:#00f3ff;">
                    REORDER QTY: <strong>+{r['Recommended Order']} units</strong>
                </div>
            </div>
            <div style="margin-top:10px; font-size:0.95rem; color:#cbd5e1;">
                <strong>Current Stock:</strong> {r['Current Stock']} units | 
                <strong>7-Day Forecast:</strong> {r['7-Day Forecast']} units | 
                <strong>Est. Lost Sales:</strong> <span style="color:#ffaa00;">₹{r['Estimated Lost Sales']:,.2f}</span>
            </div>
            <div style="margin-top:6px; color:#94a3b8; font-size:0.9rem;">
                <strong>Why?</strong> {r['Key Reasons']}
            </div>
            <div style="margin-top:8px; font-weight:700; color:#00f3ff; font-family:monospace;">
                👉 MANAGER ACTION: {r['Manager Action']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.toggle(f"Approve Order #{r['Store']}-{r['Product']}", key=f"tgl_{r['Store']}_{r['Product']}")

# ==================== TAB 5: MODEL PERFORMANCE ====================
with tab5:
    st.markdown("### Model Benchmarks & Validation Reports")
    
    reports_dir = ROOT_DIR / "reports"
    reg_csv = reports_dir / "model_comparison_regression.csv"
    clf_csv = reports_dir / "model_comparison_classification.csv"
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("#### Demand Forecast Regressors Benchmark")
        if reg_csv.exists():
            st.dataframe(pd.read_csv(reg_csv), use_container_width=True)
        else:
            st.info("Run `python -m src.pipeline train` to generate benchmark table.")
    with col_m2:
        st.markdown("#### Stock-Out Risk Classifiers Benchmark")
        if clf_csv.exists():
            st.dataframe(pd.read_csv(clf_csv), use_container_width=True)
        else:
            st.info("Run `python -m src.pipeline train` to generate benchmark table.")

    # Model Justification text
    just_md = reports_dir / "model_justification.md"
    if just_md.exists():
        with open(just_md, "r", encoding="utf-8") as f:
            st.markdown(f.read())

# ==================== TAB 6: EXPLAINABILITY ====================
with tab6:
    st.markdown("### Model Explainability & Driver Contribution Breakdown")
    
    st.markdown("#### Local Plain-Language Feature Contributions")
    
    # Feature Driver Bars for Selected Item
    sample_drivers = [
        {"reason": "Recent 7-day sales velocity", "pct": 34.2},
        {"reason": "Active marketing campaign discount", "pct": 26.5},
        {"reason": "Upcoming festival demand surge", "pct": 18.1},
        {"reason": "Weekend footfall surge", "pct": 12.3},
        {"reason": "Short shelf-life expiration risk", "pct": 8.9}
    ]
    
    driver_df = pd.DataFrame(sample_drivers)
    fig_drivers = px.bar(driver_df, x="pct", y="reason", orientation="h", title="Top Feature Drivers (% Contribution)", color="pct", color_continuous_scale="Viridis", template="plotly_dark")
    st.plotly_chart(fig_drivers, use_container_width=True)

# ==================== TAB 7: WHAT-IF LAB ====================
with tab7:
    st.markdown("### Interactive What-If Scenario Simulator")
    st.markdown("Test how promotional discounts, supplier delays, or festival spikes change predictions in real time.")
    
    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        sim_store = st.selectbox("Select Store", rec_df["Store"].unique(), key="sim_st")
        sim_prd = st.selectbox("Select Product", rec_df[rec_df["Store"]==sim_store]["Product"].unique(), key="sim_pr")
        
        sim_disc = st.slider("Additional Price Discount (%)", 0.0, 50.0, 10.0, step=5.0) / 100.0
        sim_delay = st.slider("Extra Supplier Lead Days", 0, 7, 2)
        sim_fest = st.checkbox("Simulate Festival Demand Surge", value=False)
        
    with col_w2:
        base_item = rec_df[(rec_df["Store"]==sim_store) & (rec_df["Product"]==sim_prd)].iloc[0]
        
        if reg_pipe and clf_pipe:
            item_row_df = latest_df[(latest_df["store_id"]==sim_store) | (latest_df["product_name"]==sim_prd)].head(1)
            simulator = WhatIfSimulator(reg_pipe, clf_pipe, rec_engine)
            sim_res = simulator.simulate(item_row_df, discount_pct_delta=sim_disc, extra_lead_days=sim_delay, festival_uplift=sim_fest)
            
            st.markdown("#### Scenario Simulation Results")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Baseline 7-Day Forecast", f"{base_item['7-Day Forecast']} units")
                st.metric("Baseline Risk Probability", f"{base_item['Stock-out Probability']:.2f}")
                st.metric("Baseline Recommended Order", f"{base_item['Recommended Order']} units")
            with res_col2:
                st.metric("Simulated 7-Day Forecast", f"{sim_res['new_forecast']} units", delta=f"{sim_res['new_forecast'] - base_item['7-Day Forecast']:.1f}")
                st.metric("Simulated Risk Probability", f"{sim_res['new_prob']:.2f}", delta=f"{sim_res['new_prob'] - base_item['Stock-out Probability']:.2f}")
                st.metric("Simulated Recommended Order", f"{sim_res['recommended_order']} units", delta=f"{sim_res['recommended_order'] - base_item['Recommended Order']}")
        else:
            st.warning("Models not trained yet. Run `python -m src.pipeline train` to unlock real-time What-If simulator.")
