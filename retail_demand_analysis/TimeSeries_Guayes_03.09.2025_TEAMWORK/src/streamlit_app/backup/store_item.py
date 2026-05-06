import plotly.express as px
import streamlit as st

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

st.header("🐌 Store-Item Slow Mover Analysis")

df_store_item_daily = load_table(PreDataset.STORE_ITEM_DAILY)


sold = df_store_item_daily.loc[
    df_store_item_daily["unit_sales_sum"] > 0,
    ["date", "store_nbr", "item_nbr", "unit_sales_sum"],
]

store_item_stats = sold.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
    days_sold=("date", "nunique"),
    total_units=("unit_sales_sum", "sum"),
)

store_ids = st.multiselect(
    "Highlight stores",
    sorted(store_item_stats["store_nbr"].unique()),
    default=[1],
    key="slow_mover_store_highlight",
)

store_highlight = st.selectbox(
    "Highlight store (optional)",
    options=[None, *sorted(store_item_stats["store_nbr"].unique())],
    index=0,
    key="highlight_store",
)

item_highlight = st.selectbox(
    "Highlight item (optional)",
    options=[None, *sorted(store_item_stats["item_nbr"].unique())],
    index=0,
    key="highlight_item",
)


def highlight_category(row):
    if store_highlight is not None and row["store_nbr"] == store_highlight:
        return "Selected Store"
    if item_highlight is not None and row["item_nbr"] == item_highlight:
        return "Selected Item"
    return "Other"


store_item_stats["highlight"] = store_item_stats.apply(highlight_category, axis=1)

# -----------------------------
# Interactive cutoffs
# -----------------------------
quantile = st.slider(
    "Slow mover quantile cutoff",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

days_cutoff = store_item_stats["days_sold"].quantile(quantile)
units_cutoff = store_item_stats["total_units"].quantile(quantile)

store_item_stats["slow_mover"] = (store_item_stats["days_sold"] <= days_cutoff) & (
    store_item_stats["total_units"] <= units_cutoff
)


store_item_stats["highlight_store"] = store_item_stats["store_nbr"].isin(store_ids)


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
        "Other": "#6B7280",  # grau
    },
)

fig.update_xaxes(type="log", title="Days Sold (log)")
fig.update_yaxes(type="log", title="Total Units Sold (log)")
fig.update_traces(marker={"size": 6, "opacity": 0.7})

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Metrics
# -----------------------------
st.metric(
    "Slow mover items",
    int(store_item_stats["slow_mover"].sum()),
)

st.metric(
    "Share of slow movers",
    f"{store_item_stats['slow_mover'].mean():.1%}",
)
st.markdown(
    """
    ---
    **Interpretation Guide:**
    - **Slow Movers:** Items with low sales frequency and volume.
    - **Days Sold:** Number of days an item was sold in the store.
    - **Total Units:** Total units sold for the item in the store.
    """
)
st.markdown(
    """
    ---
    _This analysis identifies slow-moving items in a selected store based on sales frequency and volume. Use these insights for inventory management and promotional strategies._
    """
)
