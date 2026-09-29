"""
Streamlit Custom Component Wrapper for Three.js 3D Store Digital Twin
Renders the rotatable, zoomable 3D supermarket aisle component.
"""

from pathlib import Path
import json
import streamlit as st
import streamlit.components.v1 as components

HTML_PATH = Path(__file__).resolve().parent / "3d_aisle_twin.html"

def render_3d_digital_twin(df_recommendations=None, height: int = 420):
    """Renders Three.js 3D Store Digital Twin component in Streamlit."""
    if not HTML_PATH.exists():
        st.error(f"3D HTML template not found at {HTML_PATH}")
        return

    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html_code = f.read()

    # Pass live DataFrame JSON into JavaScript if provided
    if df_recommendations is not None and not df_recommendations.empty:
        items = []
        for _, row in df_recommendations.head(15).iterrows():
            items.append({
                "id": f"{row.get('Store')}-{row.get('Product')}",
                "store": str(row.get("Store")),
                "product": str(row.get("Product")),
                "category": "Dairy" if "Milk" in str(row.get("Product")) else ("Bakery" if "Bread" in str(row.get("Product")) else ("Beverages" if "Soda" in str(row.get("Product")) else "Pantry")),
                "stock": int(row.get("Current Stock", 0)),
                "forecast": int(row.get("7-Day Forecast", 0)),
                "prob": float(row.get("Stock-out Probability", 0.0)),
                "risk": str(row.get("Risk"))
            })
        json_data = json.dumps(items)
        html_code = html_code.replace("let itemsData = [", f"let itemsData = {json_data}; // [")

    components.html(html_code, height=height, scrolling=False)
