import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table
from streamlit_app.components.charts import render_plotly

# CONFIGURATION

st.header("🐌 Store-Item Performance Deep Dive")

@st.cache_data(show_spinner="Analysiere Verkaufsstatistiken...")
def load_deep_dive_stats():
    df = load_table(PreDataset.STORE_ITEM_DAILY)
    # Only days with real sales for the distribution
    sold = df.loc[df["unit_sales_sum"] > 0, ["date", "store_nbr", "item_nbr", "unit_sales_sum"]]
    stats = sold.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        days_sold=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
    )
    return sold, stats

sold_raw, store_item_stats = load_deep_dive_stats()
with st.sidebar:
    st.subheader("🎯 Fokus & Definition")
    highlight_store = st.selectbox(
        "Highlight Store", 
        [None, *sorted(store_item_stats["store_nbr"].unique())],
        key="sb_store"
    )
    highlight_item = st.selectbox(
        "Highlight Item", 
        [None, *sorted(store_item_stats["item_nbr"].unique())],
        key="sb_item"
    )
    
    st.divider()
    quantile = st.slider("Slow Mover Cutoff (Quantil)", 0.01, 0.30, 0.10)

# LOGIC
# Thresholds calculation
days_cutoff = store_item_stats["days_sold"].quantile(quantile)
units_cutoff = store_item_stats["total_units"].quantile(quantile)

# Slow Mover flag
store_item_stats["slow_mover"] = (store_item_stats["days_sold"] <= days_cutoff) & \
                                 (store_item_stats["total_units"] <= units_cutoff)

# Highlight logic
store_item_stats["highlight"] = "Other"
if highlight_store:
    store_item_stats.loc[store_item_stats["store_nbr"] == highlight_store, "highlight"] = "Selected Store"
if highlight_item:
    store_item_stats.loc[store_item_stats["item_nbr"] == highlight_item, "highlight"] = "Selected Item"

# VISUALIZATION (Global View)
fig = px.scatter(
    store_item_stats,
    x="days_sold",
    y="total_units",
    color="highlight",
    symbol="slow_mover",
    hover_data=["store_nbr", "item_nbr"],
    render_mode="webgl",
    color_discrete_map={
        "Selected Store": "#58A6FF",
        "Selected Item": "#7EE787",
        "Other": "rgba(107, 114, 128, 0.2)", 
    },
    log_x=True, log_y=True
)

# Median reference lines
fig.add_vline(x=store_item_stats["days_sold"].median(), line_dash="dash", line_color="gray", annotation_text="Median")
fig.add_hline(y=store_item_stats["total_units"].median(), line_dash="dash", line_color="gray", annotation_text="Median")

# Use your central render function
render_plotly(fig)

# METRICS & DETAIL VIEW
c1, c2, c3 = st.columns(3)
c1.metric("Total Pairs", f"{len(store_item_stats):,}")
c2.metric("Slow Movers", int(store_item_stats["slow_mover"].sum()))
c3.metric("Share", f"{store_item_stats['slow_mover'].mean():.1%}")

# Detail-timeseries Visual, when selection is made
if highlight_store and highlight_item:
    st.divider()
    st.subheader(f"🔍 Detail: Store {highlight_store} - Item {highlight_item}")
    df_detail = sold_raw[(sold_raw["store_nbr"] == highlight_store) & 
                         (sold_raw["item_nbr"] == highlight_item)].sort_values("date")
    if not df_detail.empty:
        fig_line = px.line(df_detail, x="date", y="unit_sales_sum", 
                           title="Daily Sales History",
                           color_discrete_sequence=["#58A6FF"])
        render_plotly(fig_line)
    else:
        st.warning("No sales data for this combination in the selected period.")

# INTERPRETATION
st.markdown(
    """
---
### Interpretation Guide

- **Bottom-left:** Slow movers (low frequency & volume)
- **Top-right:** Core assortment drivers
- **Median lines:** Global stability reference
- **Highlight:** Compare any store or item to the global distribution

This view helps identify:
- assortment inefficiencies
- store-specific anomalies
- candidates for promotion or delisting
"""
)