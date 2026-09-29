"""
Streamlit Custom Component Wrapper for Three.js 3D Store Digital Twin & Landing Page
Renders the rotatable, zoomable 3D supermarket aisle component and cinematic landing hero.
"""

from pathlib import Path
import json
import streamlit as st
import streamlit.components.v1 as components

AISLE_HTML_PATH = Path(__file__).resolve().parent / "3d_aisle_twin.html"
LANDING_HTML_PATH = Path(__file__).resolve().parent / "landing_3d.html"

def render_3d_digital_twin(df_recommendations=None, height: int = 540):
    """Renders Three.js 3D Store Digital Twin component in Streamlit."""
    if not AISLE_HTML_PATH.exists():
        st.error(f"3D HTML template not found at {AISLE_HTML_PATH}")
        return

    with open(AISLE_HTML_PATH, "r", encoding="utf-8") as f:
        html_code = f.read()

    # Pass live DataFrame JSON into JavaScript if provided
    if df_recommendations is not None and not df_recommendations.empty:
        items = []
        for _, row in df_recommendations.head(60).iterrows():
            items.append({
                "id": f"{row.get('Store')}-{row.get('Product')}",
                "store": str(row.get("Store")),
                "store_type": "Express" if "S01" in str(row.get("Store")) else ("Supermarket" if "S02" in str(row.get("Store")) else "Hypermarket"),
                "product": str(row.get("Product")),
                "category": "Dairy" if "Milk" in str(row.get("Product")) else ("Bakery" if "Bread" in str(row.get("Product")) else ("Beverages" if "Soda" in str(row.get("Product")) else "Pantry")),
                "stock": int(row.get("Current Stock", 0)),
                "forecast": int(row.get("7-Day Forecast", 0)),
                "prob": float(row.get("Stock-out Probability", 0.0)),
                "risk": str(row.get("Risk")),
                "order": int(row.get("Recommended Order", 0)),
                "reasons": str(row.get("Key Reasons", ""))
            })
        json_data = json.dumps(items)
        html_code = html_code.replace("let itemsData = [", f"let itemsData = {json_data}; // [")

    components.html(html_code, height=height, scrolling=False)

def render_landing_3d_hero(height: int = 480):
    """Renders full-screen cinematic 3D Three.js hero animation on Landing Page."""
    if not LANDING_HTML_PATH.exists():
        st.error(f"Landing 3D HTML template not found at {LANDING_HTML_PATH}")
        return

    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        html_code = f.read()

    components.html(html_code, height=height, scrolling=False)
