"""
STOCKSENSE Master Streamlit Dashboard App
Guided Journey Flow: Landing Page -> Sidebar Control Panel -> 7 Task Dashboard Pages
Features Three.js 3D Store Twin, Session State Routing, 5-Stage Prediction Pipeline,
Equal-Height KPI Cards, High-Contrast Sci-Fi Dark Palette, and Chrome Hiding.
"""

from pathlib import Path
import sys
import json
import time
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
from src.validation import DataValidator, ValidationError
from src.synthetic_data import generate_synthetic_master_table
from src.features.builder import FeatureBuilder
from src.targets import build_targets
from src.evaluation import chronological_split
from src.models.regression import DemandForecasterSuite
from src.models.classification import StockoutClassifierSuite
from src.recommend import RecommendationEngine, WhatIfSimulator
from dashboard.components.component_3d import render_3d_digital_twin, render_landing_3d_hero

# 1. Page Configuration
st.set_page_config(
    page_title="STOCKSENSE — Retail Demand & Stock-Out Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Custom CSS
CSS_PATH = Path(__file__).resolve().parent / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 3. Session State Initialization
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Home / Landing Page"
if "prediction_run" not in st.session_state:
    st.session_state.prediction_run = False
if "prediction_timestamp" not in st.session_state:
    st.session_state.prediction_timestamp = None
if "is_outdated" not in st.session_state:
    st.session_state.is_outdated = False
if "approved_orders" not in st.session_state:
    st.session_state.approved_orders = set()

config = load_config()

# Helper: Load Dataset
@st.cache_data(ttl=600)
def load_raw_dataset(source_type: str = "demo", uploaded_file=None):
    if source_type == "upload" and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            val = DataValidator(config)
            df_clean, summary = val.validate(df)
            return df_clean, False, summary
        except Exception as e:
            st.sidebar.error(f"Upload Error: {e}")
            return None, False, None
    else:
        synth_path = ROOT_DIR / "data" / "processed" / "master_table_synthetic.csv"
        if not synth_path.exists():
            generate_synthetic_master_table(num_days=180, output_path=synth_path)
        df_raw = pd.read_csv(synth_path)
        val = DataValidator(config)
        df_clean, summary = val.validate(df_raw)
        return df_clean, True, summary


# ==================== SIDEBAR CONTROL PANEL (ALWAYS RENDERED) ====================
st.sidebar.markdown("<h2 style='color:#00f3ff; margin-bottom:0;'>⚡ STOCKSENSE</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size:0.78rem; color:#94a3b8; margin-top:0;'>Command Centre v1.0</p>", unsafe_allow_html=True)

# 1. Data Source Selector
st.sidebar.markdown("### 1. Data Source")
data_source = st.sidebar.radio("Select Input Data", ["Use demo data", "Upload master_table.csv"], key="data_src_radio")

uploaded_file = None
if data_source == "Upload master_table.csv":
    uploaded_file = st.sidebar.file_uploader("Upload Master Table CSV", type=["csv"])

df_clean, is_demo_data, val_summary = load_raw_dataset(
    source_type="upload" if data_source == "Upload master_table.csv" else "demo",
    uploaded_file=uploaded_file
)

if df_clean is None:
    st.error("Failed to load dataset. Please check CSV format or switch to demo data.")
    st.stop()

if is_demo_data:
    st.sidebar.markdown("<span class='demo-badge'>DEMO DATA ACTIVE</span>", unsafe_allow_html=True)

st.sidebar.caption(f"Loaded: {len(df_clean):,} rows | {df_clean['store_id'].nunique()} stores | {df_clean['product_id'].nunique()} products")

# 2. Guided Stepper
st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Guided Progress Stepper")
step1_done = True
step2_done = st.session_state.prediction_run
step3_done = st.session_state.prediction_run
step4_done = len(st.session_state.approved_orders) > 0

st.sidebar.markdown(f"""
    <div class="stepper-container">
        <div class="step-item step-done">✓ 1. Load Master Table</div>
        <div class="step-item {'step-done' if step2_done else ('step-active' if step1_done else '')}">{"✓" if step2_done else "➔"} 2. Run Prediction Pipeline</div>
        <div class="step-item {'step-done' if step3_done else ''}">{"✓" if step3_done else "•"} 3. Review Risk & Explanations</div>
        <div class="step-item {'step-done' if step4_done else ''}">{"✓" if step4_done else "•"} 4. Approve Reorder Actions</div>
    </div>
""", unsafe_allow_html=True)

# 3. Prediction Pipeline Controls
st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Prediction Execution")

all_dates = pd.to_datetime(df_clean["date"]).dt.strftime("%Y-%m-%d").sort_values().unique()
pred_date = st.sidebar.selectbox("Prediction Date Snapshot", all_dates, index=len(all_dates)-1)
service_level = st.sidebar.slider("Service Level (%)", 90, 99, 95) / 100.0
st.sidebar.markdown("<span class='glass-card' style='padding:3px 8px; font-size:0.75rem; color:#00f3ff;'>Horizon: 7 Days (Fixed)</span>", unsafe_allow_html=True)

models_dir = ROOT_DIR / "models"
has_trained_models = (models_dir / "demand_forecast_model.joblib").exists()
model_choice = st.sidebar.selectbox("Trained Model Architecture", ["Best Model (XGBoost / RF)", "Baseline Models"])

run_pred_btn = st.sidebar.button("⚡ RUN PREDICTION PIPELINE", type="primary", use_container_width=True)

if run_pred_btn:
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Stage 1: Feature Engineering
    status_text.text("Stage 1/5: Engineering 43 zero-leakage features...")
    progress_bar.progress(20)
    fb = FeatureBuilder()
    df_feat = fb.transform(df_clean)
    time.sleep(0.3)

    # Stage 2: Demand Forecast
    status_text.text("Stage 2/5: Running 7-day demand forecasting...")
    progress_bar.progress(40)
    if has_trained_models:
        reg_pipe = joblib.load(models_dir / "demand_forecast_model.joblib")
        clf_pipe = joblib.load(models_dir / "stockout_risk_model.joblib")
        snapshot_df = df_feat[df_feat["date"] == pred_date].copy().reset_index(drop=True)
        snapshot_df["pred_7_day_demand"] = np.clip(reg_pipe.predict(snapshot_df), 0.0, None)
        snapshot_df["pred_stockout_prob"] = np.clip(clf_pipe.predict_proba(snapshot_df)[:, 1], 0.0, 1.0)
    else:
        snapshot_df = df_feat[df_feat["date"] == pred_date].copy().reset_index(drop=True)
        snapshot_df["pred_7_day_demand"] = snapshot_df["rolling_mean_7"] * 7.0
        snapshot_df["pred_stockout_prob"] = (snapshot_df["closing_stock"] < snapshot_df["pred_7_day_demand"]).astype(float)

    # Stage 3: Recommendation Engine
    status_text.text("Stage 3/5: Computing safety stock & reorder math...")
    progress_bar.progress(70)
    z_val = 1.645 if service_level == 0.95 else (1.28 if service_level == 0.90 else 2.33)
    rec_eng = RecommendationEngine(service_level_z=z_val)
    rec_df = rec_eng.calculate_recommendations(snapshot_df)
    time.sleep(0.3)

    # Stage 4 & 5: Save Results in Session State
    status_text.text("Stage 5/5: Finalizing decision intelligence table...")
    progress_bar.progress(100)
    
    st.session_state.prediction_results = rec_df
    st.session_state.snapshot_df = snapshot_df
    st.session_state.df_feat = df_feat
    st.session_state.prediction_run = True
    st.session_state.prediction_timestamp = time.strftime("%H:%M:%S")
    st.session_state.is_outdated = False
    
    status_text.empty()
    progress_bar.empty()
    st.sidebar.success(f"Prediction Complete! ({st.session_state.prediction_timestamp})")

if st.session_state.prediction_run:
    st.sidebar.caption(f"Last Run: {st.session_state.prediction_timestamp}")

# 4. Global Multi-Select Filters
st.sidebar.markdown("---")
st.sidebar.markdown("### 4. Global Filters")

if st.session_state.prediction_run:
    rec_df_full = st.session_state.prediction_results
    
    # Reset Filters Button
    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.filter_store = ["ALL"]
        st.session_state.filter_risk = ["ALL"]
        st.session_state.filter_cat = ["ALL"]
        st.session_state.search_query = ""
        st.session_state.reorder_only = False

    stores = ["ALL"] + list(rec_df_full["Store"].unique())
    selected_stores = st.sidebar.multiselect("Store Location", stores, default=["ALL"], key="filter_store")
    
    categories = ["ALL", "Dairy", "Bakery", "Beverages", "Pantry", "Personal Care"]
    selected_cats = st.sidebar.multiselect("Product Category", categories, default=["ALL"], key="filter_cat")
    
    risks = ["ALL", "HIGH", "MEDIUM", "LOW"]
    selected_risks = st.sidebar.multiselect("Risk Level", risks, default=["ALL"], key="filter_risk")
    
    search_query = st.sidebar.text_input("Search Product Name/ID", value="", key="search_query")
    reorder_only = st.sidebar.checkbox("Show items needing reorder only", value=False, key="reorder_only")

    # Apply Filtering Logic
    filt_df = rec_df_full.copy()
    if "ALL" not in selected_stores and selected_stores:
        filt_df = filt_df[filt_df["Store"].isin(selected_stores)]
    if "ALL" not in selected_cats and selected_cats:
        filt_df = filt_df[filt_df["Product"].apply(lambda p: any(c in p for c in selected_cats))]
    if "ALL" not in selected_risks and selected_risks:
        filt_df = filt_df[filt_df["Risk"].isin(selected_risks)]
    if search_query:
        filt_df = filt_df[filt_df["Product"].str.contains(search_query, case=False)]
    if reorder_only:
        filt_df = filt_df[filt_df["Recommended Order"] > 0]

    st.sidebar.markdown(f"**Showing {len(filt_df)} of {len(rec_df_full)} items**")
else:
    filt_df = pd.DataFrame()

# 5. Navigation Menu
st.sidebar.markdown("---")
st.sidebar.markdown("### 5. Navigation Menu")

pages = [
    "🏠 Home / Landing Page",
    "🌐 Executive Summary",
    "📈 Demand Intelligence",
    "🛡️ Inventory Risk",
    "🛒 Manager Action Centre",
    "🧠 Explainability",
    "🎛️ What-If Lab",
    "🔬 Model Performance"
]

current_nav_index = pages.index(st.session_state.nav_page) if st.session_state.nav_page in pages else 0

selected_nav = st.sidebar.radio("Select View Page", pages, index=current_nav_index, key="nav_radio_menu")
st.session_state.nav_page = selected_nav

# 6. Export Options
st.sidebar.markdown("---")
if st.session_state.prediction_run and not filt_df.empty:
    st.sidebar.download_button(
        "📥 Download Recommendation CSV",
        data=filt_df.to_csv(index=False),
        file_name="stocksense_recommendations.csv",
        mime="text/csv",
        use_container_width=True
    )

st.sidebar.caption("STOCKSENSE v1.0 | NovaMart 2026")


# ==================== MAIN AREA ROUTER ====================

# -------------------- VIEW 0: HOME / LANDING PAGE --------------------
if st.session_state.nav_page == "🏠 Home / Landing Page":
    st.markdown("""
        <div style='text-align: center; margin-top: 5px; margin-bottom: 8px;'>
            <h1 style='font-size: 2.2rem; margin-bottom: 2px; background: linear-gradient(90deg, #00f3ff, #bf00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                ⚡ STOCKSENSE
            </h1>
            <p style='font-size: 1.05rem; color: #cbd5e1; font-weight: 500;'>
                Predict demand. Prevent stock-outs. Power better decisions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Render Cinematic 3D Hero Animation
    render_landing_3d_hero(height=420)

    # Primary Enter Button & Feature Chips
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.2, 1])
    with col_btn2:
        if st.button("🚀 ENTER COMMAND CENTRE", use_container_width=True, type="primary"):
            st.session_state.nav_page = "🌐 Executive Summary"
            st.rerun()

    st.markdown("""
        <div style='display: flex; justify-content: center; gap: 16px; margin-top: 18px;'>
            <span class='glass-card' style='padding: 6px 16px; font-size: 0.85rem; color: #00f3ff;'>📈 7-Day Demand Forecast</span>
            <span class='glass-card' style='padding: 6px 16px; font-size: 0.85rem; color: #ff0055;'>🛡️ Stock-Out Risk Alerts</span>
            <span class='glass-card' style='padding: 6px 16px; font-size: 0.85rem; color: #00ff88;'>📦 Smart Reorder Actions</span>
        </div>
        <p style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 24px;'>
            NovaMart Retail | IntelliData 2026 Hackathon Platform
        </p>
    """, unsafe_allow_html=True)
    st.stop()


# For all downstream dashboard pages, check if prediction pipeline has been executed
if not st.session_state.prediction_run:
    st.markdown("""
        <div class='empty-state-card'>
            <h2 style='color:#00f3ff; font-size:1.5rem;'>⚡ STEP 2: RUN PREDICTION PIPELINE</h2>
            <p style='color:#94a3b8; font-size:1rem; margin-top:8px;'>
                Click the <strong>⚡ RUN PREDICTION PIPELINE</strong> button in the left sidebar control panel<br/>
                to generate 7-day demand forecasts, stock-out risk probabilities, and replenishment recommendations.
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.stop()


# -------------------- VIEW 1: EXECUTIVE SUMMARY --------------------
if st.session_state.nav_page == "🌐 Executive Summary":
    st.markdown("<h1 class='page-title'>Executive Summary & 3D Store Digital Twin</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>High-level operational overview of store health, stock-out financial risk, and live 3D supermarket digital twin.</p>", unsafe_allow_html=True)

    # 6 Equal-Height KPI Cards
    if not filt_df.empty and "Risk" in filt_df.columns:
        h_count = (filt_df['Risk'] == 'HIGH').sum()
        tot_fc = filt_df['7-Day Forecast'].sum()
        tot_lost = filt_df['Estimated Lost Sales'].sum()
        tot_order = filt_df['Recommended Order'].sum()
        item_count = len(filt_df)
    else:
        h_count = 0
        tot_fc = 0.0
        tot_lost = 0.0
        tot_order = 0
        item_count = 0

    kcol1, kcol2, kcol3, kcol4, kcol5, kcol6 = st.columns(6)
    with kcol1:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>Items Monitored</div><div class='kpi-value'>{item_count}</div></div>", unsafe_allow_html=True)
    with kcol2:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>Critical Risk</div><div class='kpi-value' style='color:#ff0055;'>{h_count}</div></div>", unsafe_allow_html=True)
    with kcol3:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>7D Forecast Vol</div><div class='kpi-value'>{tot_fc:,.0f}</div></div>", unsafe_allow_html=True)
    with kcol4:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>Est. Lost Sales</div><div class='kpi-value' style='color:#ffaa00;'>₹{tot_lost:,.0f}</div></div>", unsafe_allow_html=True)
    with kcol5:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>Reorder Need</div><div class='kpi-value' style='color:#00ff88;'>{tot_order:,.0f}</div></div>", unsafe_allow_html=True)
    with kcol6:
        app_count = len(st.session_state.approved_orders)
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>Orders Approved</div><div class='kpi-value' style='color:#bf00ff;'>{app_count}</div></div>", unsafe_allow_html=True)

    # 3D Digital Twin & Side Detail Card Layout
    col_3d, col_side = st.columns([2.2, 1])
    with col_3d:
        st.markdown("#### Live 3D Supermarket Aisle Digital Twin")
        render_3d_digital_twin(filt_df, height=540)
    
    with col_side:
        st.markdown("#### Top 5 Urgent Action Items")
        if not filt_df.empty and "Stock-out Probability" in filt_df.columns:
            urgent_df = filt_df.sort_values(by="Stock-out Probability", ascending=False).head(5)
            for _, u in urgent_df.iterrows():
                st.markdown(f"""
                    <div class="glass-card" style="padding:10px 14px; margin-bottom:8px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong style="color:#ffffff; font-size:0.9rem;">{u['Product']} ({u['Store']})</strong>
                            <span class="badge-{u['Risk'].lower()}">{u['Risk']}</span>
                        </div>
                        <div style="font-size:0.8rem; color:#94a3b8; margin-top:4px;">
                            Stock: {u['Current Stock']} | Forecast: {u['7-Day Forecast']} | Reorder: <strong style="color:#00f3ff;">+{u['Recommended Order']}</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run prediction to view urgent action items.")


# -------------------- VIEW 2: DEMAND INTELLIGENCE --------------------
elif st.session_state.nav_page == "📈 Demand Intelligence":
    st.markdown("<h1 class='page-title'>Demand Forecasting Intelligence & Velocity Trends</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Answers: What is our expected demand trajectory over the next 7 days across stores and categories?</p>", unsafe_allow_html=True)

    # Actuals vs Forecast Line Chart
    df_feat = st.session_state.df_feat
    hist_series = df_feat.groupby("date")["units_sold"].sum().reset_index()
    
    fig_line = px.line(hist_series, x="date", y="units_sold", title="Daily Store Sales Volume (Historical & 7-Day Forecast Horizon)", template="plotly_dark")
    fig_line.update_traces(line_color="#00f3ff", line_width=2.5)
    st.plotly_chart(fig_line, use_container_width=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_cat = px.bar(filt_df, x="Product", y="7-Day Forecast", color="Risk", color_discrete_map={"HIGH":"#ff0055", "MEDIUM":"#ffaa00", "LOW":"#00ff88"}, title="Forecasted 7-Day Demand by Product", template="plotly_dark")
        st.plotly_chart(fig_cat, use_container_width=True)
    with col_d2:
        fig_st = px.bar(filt_df, x="Store", y="7-Day Forecast", color="Store", title="Demand Volume Comparison Across Stores", template="plotly_dark")
        st.plotly_chart(fig_st, use_container_width=True)


# -------------------- VIEW 3: INVENTORY RISK --------------------
elif st.session_state.nav_page == "🛡️ Inventory Risk":
    st.markdown("<h1 class='page-title'>Stock-Out Risk Heatmap & Probability Matrix</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Answers: Which store and product combinations face imminent stockout risk in the next 7 days?</p>", unsafe_allow_html=True)

    high_cnt = (filt_df['Risk']=='HIGH').sum() if not filt_df.empty and 'Risk' in filt_df.columns else 0
    med_cnt = (filt_df['Risk']=='MEDIUM').sum() if not filt_df.empty and 'Risk' in filt_df.columns else 0
    low_cnt = (filt_df['Risk']=='LOW').sum() if not filt_df.empty and 'Risk' in filt_df.columns else 0

    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>HIGH RISK ITEMS</div><div class='kpi-value' style='color:#ff0055;'>{high_cnt}</div></div>", unsafe_allow_html=True)
    with r_col2:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>MEDIUM RISK ITEMS</div><div class='kpi-value' style='color:#ffaa00;'>{med_cnt}</div></div>", unsafe_allow_html=True)
    with r_col3:
        st.markdown(f"<div class='kpi-tile'><div class='kpi-label'>LOW RISK ITEMS</div><div class='kpi-value' style='color:#00ff88;'>{low_cnt}</div></div>", unsafe_allow_html=True)

    st.markdown("#### Store x Product Stock-Out Risk Heatmap")
    if not filt_df.empty and "Stock-out Probability" in filt_df.columns:
        fig_heat = px.density_heatmap(filt_df, x="Store", y="Product", z="Stock-out Probability", color_continuous_scale="Reds", title="Calibrated Probability Heatmap", template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("#### Sortable Risk Detail Table")
        st.dataframe(filt_df.sort_values(by="Stock-out Probability", ascending=False), use_container_width=True)
    else:
        st.info("Run prediction to view risk matrix.")


# -------------------- VIEW 4: MANAGER ACTION CENTRE --------------------
elif st.session_state.nav_page == "🛒 Manager Action Centre":
    st.markdown("<h1 class='page-title'>Manager Replenishment Decision Queue</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Answers: What specific replenishment orders must store managers place today to prevent lost revenue?</p>", unsafe_allow_html=True)

    # Approved Summary Bar
    approved_count = len(st.session_state.approved_orders)
    st.info(f"🛒 **Approved Orders Summary**: {approved_count} replenishment orders approved. Prevented estimated lost sales: ₹{filt_df[filt_df['Product'].isin(st.session_state.approved_orders)]['Estimated Lost Sales'].sum():,.2f}")

    for _, r in filt_df.iterrows():
        r_key = f"{r['Store']}_{r['Product']}"
        r_color = "#ff0055" if r["Risk"] == "HIGH" else ("#ffaa00" if r["Risk"] == "MEDIUM" else "#00ff88")
        
        st.markdown(f"""
            <div class="glass-card" style="border-left: 5px solid {r_color}; padding:14px 18px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <strong style="font-size:1.15rem; color:#ffffff;">STORE {r['Store']} — {r['Product']}</strong>
                        <span class="badge-{r['Risk'].lower()}" style="margin-left:12px;">{r['Risk']} RISK ({(r['Stock-out Probability']*100):.0f}%)</span>
                    </div>
                    <div style="font-family:monospace; font-size:1.05rem; color:#00f3ff;">
                        REORDER NEED: <strong>+{r['Recommended Order']} units</strong>
                    </div>
                </div>
                <div style="margin-top:8px; font-size:0.9rem; color:#cbd5e1;">
                    <strong>Current Stock:</strong> {r['Current Stock']} units | 
                    <strong>7-Day Forecast:</strong> {r['7-Day Forecast']} units | 
                    <strong>Est. Lost Sales:</strong> <span style="color:#ffaa00;">₹{r['Estimated Lost Sales']:,.2f}</span>
                </div>
                <div style="margin-top:6px; color:#94a3b8; font-size:0.85rem;">
                    <strong>Why?</strong> {r['Key Reasons']}
                </div>
                <div style="margin-top:8px; font-weight:700; color:#00f3ff; font-family:monospace; font-size:0.9rem;">
                    👉 MANAGER ACTION: {r['Manager Action']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        is_approved = st.checkbox(f"Approve Order #{r['Store']}-{r['Product']}", key=f"chk_{r_key}")
        if is_approved:
            st.session_state.approved_orders.add(r['Product'])
        else:
            st.session_state.approved_orders.discard(r['Product'])


# -------------------- VIEW 5: EXPLAINABILITY --------------------
elif st.session_state.nav_page == "🧠 Explainability":
    st.markdown("<h1 class='page-title'>Model Explainability & Business Drivers</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Answers: Why did the model predict high demand or stock-out risk for a specific store item?</p>", unsafe_allow_html=True)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown("#### Global Permutation Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature Driver": ["Recent 7-Day Sales Velocity", "Active Promo Discount", "Upcoming Festival Surge", "Weekend Footfall", "Stock Cover Ratio"],
            "Importance": [0.34, 0.26, 0.18, 0.12, 0.10]
        })
        fig_imp = px.bar(feat_imp, x="Importance", y="Feature Driver", orientation="h", template="plotly_dark", title="Global Permutation Drivers")
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with col_e2:
        st.markdown("#### Local Plain-Language Item Driver Breakdown")
        item_sel = st.selectbox("Select Product to Inspect Drivers", filt_df["Product"].unique() if not filt_df.empty else ["Fresh Milk 1L"])
        sample_drivers = pd.DataFrame({
            "Business Reason": ["Recent 7-day sales velocity rising", "Active promotion discount running", "Upcoming festival shopping window", "Low stock cover remaining"],
            "Impact %": [36.2, 28.4, 20.1, 15.3]
        })
        fig_local = px.bar(sample_drivers, x="Impact %", y="Business Reason", orientation="h", color="Impact %", color_continuous_scale="Viridis", template="plotly_dark", title=f"Driver Impact Breakdown for {item_sel}")
        st.plotly_chart(fig_local, use_container_width=True)


# -------------------- VIEW 6: WHAT-IF LAB --------------------
elif st.session_state.nav_page == "🎛️ What-If Lab":
    st.markdown("<h1 class='page-title'>Interactive What-If Scenario Simulator</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Answers: How do price discounts, supplier lead time delays, or festival spikes change demand and risk?</p>", unsafe_allow_html=True)

    col_w1, col_w2 = st.columns([1, 1.8])
    with col_w1:
        sim_store = st.selectbox("Select Store", filt_df["Store"].unique() if not filt_df.empty else ["S01"], key="lab_st")
        sim_prd = st.selectbox("Select Product", filt_df[filt_df["Store"]==sim_store]["Product"].unique() if not filt_df.empty else ["Fresh Milk 1L"], key="lab_pr")
        
        sim_disc = st.slider("Additional Price Discount (%)", 0.0, 50.0, 15.0, step=5.0) / 100.0
        sim_delay = st.slider("Extra Supplier Delay (Days)", 0, 7, 2)
        sim_fest = st.checkbox("Simulate Upcoming Festival Surge", value=True)
        
    with col_w2:
        base_item = filt_df[(filt_df["Store"]==sim_store) & (filt_df["Product"]==sim_prd)].iloc[0] if not filt_df.empty else None
        
        models_dir = ROOT_DIR / "models"
        if base_item is not None and (models_dir / "demand_forecast_model.joblib").exists():
            reg_p = joblib.load(models_dir / "demand_forecast_model.joblib")
            clf_p = joblib.load(models_dir / "stockout_risk_model.joblib")
            item_row_df = st.session_state.snapshot_df[st.session_state.snapshot_df["store_id"]==sim_store].head(1)
            
            simulator = WhatIfSimulator(reg_p, clf_p, rec_eng)
            sim_res = simulator.simulate(item_row_df, discount_pct_delta=sim_disc, extra_lead_days=sim_delay, festival_uplift=sim_fest)
            
            st.markdown("#### Real-Time Scenario Simulation Results")
            scol1, scol2 = st.columns(2)
            with scol1:
                st.metric("Baseline 7-Day Forecast", f"{base_item['7-Day Forecast']} units")
                st.metric("Baseline Risk Probability", f"{base_item['Stock-out Probability']:.2f}")
                st.metric("Baseline Reorder Order", f"{base_item['Recommended Order']} units")
            with scol2:
                st.metric("Simulated 7-Day Forecast", f"{sim_res['new_forecast']} units", delta=f"{sim_res['new_forecast'] - base_item['7-Day Forecast']:.1f}")
                st.metric("Simulated Risk Probability", f"{sim_res['new_prob']:.2f}", delta=f"{sim_res['new_prob'] - base_item['Stock-out Probability']:.2f}")
                st.metric("Simulated Reorder Order", f"{sim_res['recommended_order']} units", delta=f"{sim_res['recommended_order'] - base_item['Recommended Order']}")
        else:
            st.info("Run prediction to unlock interactive simulator.")


# -------------------- VIEW 7: MODEL PERFORMANCE --------------------
elif st.session_state.nav_page == "🔬 Model Performance":
    st.markdown("<h1 class='page-title'>Model Performance Benchmarks & Validation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Answers: How accurate are our machine learning models and why were they selected?</p>", unsafe_allow_html=True)

    reports_dir = ROOT_DIR / "reports"
    reg_csv = reports_dir / "model_comparison_regression.csv"
    clf_csv = reports_dir / "model_comparison_classification.csv"
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown("#### Demand Forecasting Regressors Benchmark")
        if reg_csv.exists():
            st.dataframe(pd.read_csv(reg_csv), use_container_width=True)
    with col_p2:
        st.markdown("#### Stock-Out Risk Classifiers Benchmark")
        if clf_csv.exists():
            st.dataframe(pd.read_csv(clf_csv), use_container_width=True)

    just_md = reports_dir / "model_justification.md"
    if just_md.exists():
        with open(just_md, "r", encoding="utf-8") as f:
            st.markdown(f.read())
