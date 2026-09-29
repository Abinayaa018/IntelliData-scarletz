"""
Streamlit Custom Component Wrapper for Three.js 3D Store Digital Twin & Landing Page

Renders the rotatable, zoomable 3D supermarket aisle component and cinematic landing hero.
"""

from pathlib import Path
import json

import streamlit as st
import streamlit.components.v1 as components


# Paths to the HTML files
AISLE_HTML_PATH = Path(__file__).resolve().parent / "3d_aisle_twin.html"
LANDING_HTML_PATH = Path(__file__).resolve().parent / "landing_3d.html"


def render_3d_digital_twin(df_recommendations=None, height: int = 540):
    """Renders Three.js 3D Store Digital Twin component in Streamlit."""

    # Check whether the HTML file exists
    if not AISLE_HTML_PATH.exists():
        st.error(
            f"3D HTML template not found at {AISLE_HTML_PATH}"
        )
        return

    # Read HTML
    with open(AISLE_HTML_PATH, "r", encoding="utf-8") as f:
        html_code = f.read()

    # Pass live DataFrame JSON into JavaScript if provided
    if df_recommendations is not None and not df_recommendations.empty:

        items = []

        for _, row in df_recommendations.head(60).iterrows():

            store = str(row.get("Store", "Unknown"))
            product = str(row.get("Product", "Unknown"))

            # Determine store type
            if "S01" in store:
                store_type = "Express"
            elif "S02" in store:
                store_type = "Supermarket"
            else:
                store_type = "Hypermarket"

            # Determine category
            if "Milk" in product:
                category = "Dairy"
            elif "Bread" in product:
                category = "Bakery"
            elif "Soda" in product:
                category = "Beverages"
            else:
                category = "Pantry"

            # Safely convert numerical values
            try:
                stock = int(row.get("Current Stock", 0))
            except (ValueError, TypeError):
                stock = 0

            try:
                forecast = int(row.get("7-Day Forecast", 0))
            except (ValueError, TypeError):
                forecast = 0

            try:
                prob = float(row.get("Stock-out Probability", 0.0))
            except (ValueError, TypeError):
                prob = 0.0

            try:
                order = int(row.get("Recommended Order", 0))
            except (ValueError, TypeError):
                order = 0

            items.append(
                {
                    "id": f"{store}-{product}",
                    "store": store,
                    "store_type": store_type,
                    "product": product,
                    "category": category,
                    "stock": stock,
                    "forecast": forecast,
                    "prob": prob,
                    "risk": str(row.get("Risk", "Unknown")),
                    "order": order,
                    "reasons": str(row.get("Key Reasons", "")),
                }
            )

        # Convert Python data to JSON for JavaScript
        json_data = json.dumps(items)

        # Replace default JavaScript data
        html_code = html_code.replace(
            "let itemsData = [",
            f"let itemsData = {json_data}; // ["
        )

    # Render Three.js HTML inside Streamlit
    components.html(
        html_code,
        height=height,
        scrolling=False
    )


def render_landing_3d_hero(height: int = 480):
    """Renders full-screen cinematic 3D Three.js hero animation on Landing Page."""

    # Check whether the HTML file exists
    if not LANDING_HTML_PATH.exists():
        st.error(
            f"Landing 3D HTML template not found at {LANDING_HTML_PATH}"
        )
        return

    # Read HTML
    with open(LANDING_HTML_PATH, "r", encoding="utf-8") as f:
        html_code = f.read()

    # Render landing page
    components.html(
        html_code,
        height=height,
        scrolling=False
    )
