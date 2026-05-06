import plotly.express as px
import streamlit as st

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Store-Item Deep Dive", layout="wide")
st.header("🐌 Store-Item Sales Performance (Global View)")


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare():
    df = load_table(PreDataset.STORE_ITEM_DAILY)

    sold = df.loc[
        df["unit_sales_sum"] > 0,
        ["date", "store_nbr", "item_nbr", "unit_sales_sum"],
    ]

    stats = sold.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        days_sold=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
    )

    return sold, stats


sold, store_item_stats = load_and_prepare()

# ------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------
with st.sidebar:
    st.subheader("🎯 Highlight")
    highlight_store = st.selectbox(
        "Highlight store",
        options=[None, *sorted(store_item_stats["store_nbr"].unique())],
        index=0,
        key="highlight_store",
    )

    highlight_item = st.selectbox(
        "Highlight item",
        options=[None, *sorted(store_item_stats["item_nbr"].unique())],
        index=0,
        key="highlight_item",
    )

    st.divider()
    st.subheader("🐌 Slow mover definition")

    quantile = st.slider(
        "Quantile cutoff",
        min_value=0.01,
        max_value=0.30,
        value=0.10,
        step=0.01,
    )

# ------------------------------------------------------------------
# SLOW MOVER LOGIC
# ------------------------------------------------------------------
days_cutoff, units_cutoff = store_item_stats[["days_sold", "total_units"]].quantile(
    quantile
)

store_item_stats["slow_mover"] = (store_item_stats["days_sold"] <= days_cutoff) & (
    store_item_stats["total_units"] <= units_cutoff
)

# ------------------------------------------------------------------
# HIGHLIGHT LOGIC (FAST / VECTORIZED)
# ------------------------------------------------------------------
store_item_stats["highlight"] = "Other"

if highlight_store is not None:
    store_item_stats.loc[
        store_item_stats["store_nbr"] == highlight_store,
        "highlight",
    ] = "Selected Store"

if highlight_item is not None:
    store_item_stats.loc[
        store_item_stats["item_nbr"] == highlight_item,
        "highlight",
    ] = "Selected Item"

# ------------------------------------------------------------------
# MEDIANS (GLOBAL REFERENCE)
# ------------------------------------------------------------------
median_days = store_item_stats["days_sold"].median()
median_units = store_item_stats["total_units"].median()

# ------------------------------------------------------------------
# GLOBAL SCATTER
# ------------------------------------------------------------------
fig = px.scatter(
    store_item_stats,
    x="days_sold",
    y="total_units",
    color="highlight",
    symbol="slow_mover",
    hover_data=["store_nbr", "item_nbr"],
    title="Global Store-Item Sales Performance",
    render_mode="webgl",
    color_discrete_map={
        "Selected Store": "#58A6FF",
        "Selected Item": "#7EE787",
        "Other": "#6B7280",
    },
)

# Median reference lines
fig.add_vline(
    x=median_days,
    line_dash="dash",
    line_color="gray",
    annotation_text="Median days sold",
)

fig.add_hline(
    y=median_units,
    line_dash="dash",
    line_color="gray",
    annotation_text="Median total units",
)

fig.update_xaxes(type="log", title="Days Sold (log)")
fig.update_yaxes(type="log", title="Total Units Sold (log)")
fig.update_traces(marker={"size": 6, "opacity": 0.65})

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric(
    "Total store-item pairs",
    f"{len(store_item_stats):,}",
)

c2.metric(
    "Slow movers",
    int(store_item_stats["slow_mover"].sum()),
)

c3.metric(
    "Slow mover share",
    f"{store_item_stats['slow_mover'].mean():.1%}",
)

# ------------------------------------------------------------------
# DETAIL VIEW (CLICK-BASED)
# ------------------------------------------------------------------
st.divider()
st.subheader("🔍 Detail View")

clicked = st.plotly_chart(
    fig,
    use_container_width=True,
    key="scatter_click",
)

selected_store = highlight_store
selected_item = highlight_item

if selected_store is not None and selected_item is not None:
    df_detail = sold[
        (sold["store_nbr"] == selected_store) & (sold["item_nbr"] == selected_item)
    ]

    if not df_detail.empty:
        fig_detail = px.line(
            df_detail,
            x="date",
            y="unit_sales_sum",
            title=f"Store {selected_store} - Item {selected_item} Daily Sales",
        )

        st.plotly_chart(fig_detail, use_container_width=True)
    else:
        st.info("No sales data available for this store-item pair.")

# ------------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------------
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
