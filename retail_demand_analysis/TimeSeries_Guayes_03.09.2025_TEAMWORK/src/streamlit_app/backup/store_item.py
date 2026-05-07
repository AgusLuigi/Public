import sys
from pathlib import Path
import numpy as np
import plotly.express as px
import streamlit as st

# 1. PFAD-LOGIK (Chirurgisch hinzugefügt)
# Wir gehen 3 Ebenen hoch: pages -> streamlit_app -> src -> ROOT
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table
from streamlit_app.components.charts import render_plotly

st.header("🐌 Store-Item Slow Mover Analysis")

# DATA LOADING (Mit Caching für bessere Performance)
@st.cache_data(show_spinner="Lade Verkaufsdaten...")
def get_store_item_stats():
    df = load_table(PreDataset.STORE_ITEM_DAILY)
    
    # Nur Tage mit Verkäufen betrachten
    sold = df.loc[
        df["unit_sales_sum"] > 0,
        ["date", "store_nbr", "item_nbr", "unit_sales_sum"],
    ]

    stats = sold.groupby(["store_nbr", "item_nbr"], as_index=False).agg(days_sold=("date", "nunique"),total_units=("unit_sales_sum", "sum"),)
    return stats

store_item_stats = get_store_item_stats()

# SIDEBAR / FILTER LOGIK
with st.sidebar:
    st.subheader("🎯 Highlight & Cutoff")
    
    store_highlight = st.selectbox(
        "Highlight Store",
        options=[None, *sorted(store_item_stats["store_nbr"].unique())],
        index=0,
        key="highlight_store",
    )

    item_highlight = st.selectbox(
        "Highlight Item",
        options=[None, *sorted(store_item_stats["item_nbr"].unique())],
        index=0,
        key="highlight_item",
    )

    st.divider()
    
    quantile = st.slider(
        "Slow mover quantile cutoff",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
    )

# LOGIK: BERECHNUNGEN (Chirurgisch optimiert)
# 1. Interactive cutoffs (Zuerst die Inputs abfragen)
quantile = st.slider(
    "Slow mover quantile cutoff",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

# 2. Schwellenwerte berechnen
days_cutoff = store_item_stats["days_sold"].quantile(quantile)
units_cutoff = store_item_stats["total_units"].quantile(quantile)

# 3. Slow Mover Flag (Vektorisierte Berechnung)
store_item_stats["slow_mover"] = (store_item_stats["days_sold"] <= days_cutoff) & \
                                 (store_item_stats["total_units"] <= units_cutoff)

# 4. Highlight-Kategorien (Vektorisierung statt .apply für maximale Geschwindigkeit)
store_item_stats["highlight"] = "Other"

if store_highlight is not None:
    store_item_stats.loc[store_item_stats["store_nbr"] == store_highlight, "highlight"] = "Selected Store"

if item_highlight is not None:
    store_item_stats.loc[store_item_stats["item_nbr"] == item_highlight, "highlight"] = "Selected Item"

# VISUALIZATION (Plotly)
fig = px.scatter(
    store_item_stats,
    x="days_sold",
    y="total_units",
    color="highlight",
    hover_data=["store_nbr", "item_nbr"],
    title="Store-Item Sales Performance (Global)",
    color_discrete_map={
        "Selected Store": "#58A6FF",  # blau
        "Selected Item": "#7EE787",  # grün
        "Other": "rgba(107, 114, 128, 0.4)",  # grau mit Transparenz für bessere Sichtbarkeit
    },
)

fig.update_xaxes(type="log", title="Days Sold (log)")
fig.update_yaxes(type="log", title="Total Units Sold (log)")
fig.update_traces(marker={"size": 6, "opacity": 0.7})

st.plotly_chart(fig, use_container_width=True)

# METRICS & GUIDE
c1, c2 = st.columns(2)
c1.metric("Slow mover items", int(store_item_stats["slow_mover"].sum()))
c2.metric("Share of slow movers", f"{store_item_stats['slow_mover'].mean():.1%}")

st.markdown(
    """
    **Interpretation Guide:**
    - **Slow Movers:** Items with low sales frequency and volume.
    - **Days Sold:** Number of days an item was sold in the store.
    - **Total Units:** Total units sold for the item in the store.
    
    ---
    _This analysis identifies slow-moving items in a selected store based on sales frequency and volume. Use these insights for inventory management and promotional strategies._
    """
)