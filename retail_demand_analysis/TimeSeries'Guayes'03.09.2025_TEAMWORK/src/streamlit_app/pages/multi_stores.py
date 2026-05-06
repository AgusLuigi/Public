import plotly.express as px
import streamlit as st

from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.date_utils import normalize_time_col
from Favorita_TSA.utils.preprocess_data import load_table
from streamlit_app.components.charts import render_plotly

st.title("Store Analysis")

df_store_daily = load_table(PreDataset.STORE_DAILY)
df_store_weekly = load_table(PreDataset.STORE_WEEKLY)
df_store_monthly = load_table(PreDataset.STORE_MONTHLY)

if "week_start" in df_store_weekly.columns:
    df_store_weekly = normalize_time_col(df_store_weekly, "week_start")
    df_store_weekly["week_ts"] = df_store_weekly["week_start"]
elif "week" in df_store_weekly.columns:
    df_store_weekly = normalize_time_col(df_store_weekly, "week")
    df_store_weekly["week_ts"] = df_store_weekly["week"]
else:
    raise ValueError("No week column found (expected 'week' or 'week_start')")


store_ids = st.multiselect(
    "Select stores",
    options=sorted(df_store_daily["store_nbr"].unique()),
    default=[cfg.ui.default_store],
)

if not store_ids:
    st.info("Please select at least one store.")
    st.stop()

if len(store_ids) > 10:
    st.warning("Please select at most 10 stores for readability.")


df_store_daily_choice = df_store_daily[df_store_daily["store_nbr"].isin(store_ids)]
df_store_weekly_choice = df_store_weekly[df_store_weekly["store_nbr"].isin(store_ids)]
df_store_monthly_choice = df_store_monthly[
    df_store_monthly["store_nbr"].isin(store_ids)
]

st.header("Trend")
st.caption("Long-term sales trends indicate growth or decline.")

# Plotting Monthly
fig_monthly = px.line(
    df_store_monthly_choice,
    x="month",
    y="unit_sales_sum",
    color="store_nbr",
    title="Monthly Sales",
)
render_plotly(fig_monthly)


# Plotting Weekly
fig_weekly = px.line(
    df_store_weekly_choice,
    x="week",
    y="unit_sales_sum",
    color="store_nbr",
    title="Weekly Sales",
)
render_plotly(fig_weekly)


# Plotting Daily
fig_daily = px.line(
    df_store_daily_choice,
    x="date",
    y="unit_sales_sum",
    color="store_nbr",
    title="Daily Sales",
)
render_plotly(fig_daily)

st.header("Store Stability")

st.caption("How stable are sales over time? Lower variability means higher stability.")

stability = df_store_weekly_choice.groupby("store_nbr", as_index=False).agg(
    mean_sales=("unit_sales_sum", "mean"),
    std_sales=("unit_sales_sum", "std"),
)

stability["cv"] = stability["std_sales"] / stability["mean_sales"]

st.dataframe(
    stability.sort_values("cv"),
    use_container_width=True,
)

fig_stability = px.bar(
    stability,
    x="store_nbr",
    y="cv",
    color="store_nbr",
    title="Sales Variability (Coefficient of Variation)",
)
render_plotly(fig_stability)


st.header("Seasonality")

st.caption("Weekly sales patterns reveal recurring customer behavior.")

df_store_daily_choice["dow"] = df_store_daily_choice["date"].dt.day_name()

seasonality = df_store_daily_choice.groupby(["store_nbr", "dow"], as_index=False).agg(
    mean_sales=("unit_sales_sum", "mean")
)

fig_seasonality = px.bar(
    seasonality,
    x="dow",
    y="mean_sales",
    color="store_nbr",
    category_orders={
        "dow": [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
    },
    title="Average Sales by Day of Week",
)
render_plotly(fig_seasonality)

st.header("Outliers")

st.caption(
    "Extreme deviations from the rolling average may indicate promotions or anomalies."
)

df_outliers = df_store_weekly_choice.copy()
df_outliers = df_outliers.sort_values("week")

_rw = cfg.analysis.rolling_window
_rp = cfg.analysis.rolling_min_periods
df_outliers["rolling_mean"] = df_outliers.groupby("store_nbr")[
    "unit_sales_sum"
].transform(lambda s: s.rolling(_rw, min_periods=_rp).mean())

df_outliers["rolling_std"] = df_outliers.groupby("store_nbr")[
    "unit_sales_sum"
].transform(lambda s: s.rolling(_rw, min_periods=_rp).std())

df_outliers["z_score"] = (
    df_outliers["unit_sales_sum"] - df_outliers["rolling_mean"]
) / df_outliers["rolling_std"]

_z = cfg.analysis.zscore_threshold
fig_outliers = px.scatter(
    df_outliers,
    x="week",
    y="unit_sales_sum",
    color=(df_outliers["z_score"].abs() > _z),
    title=f"Weekly Sales with Outliers (|Z| > {_z})",
)
render_plotly(fig_outliers)


st.markdown(
    """
    ---
    **Interpretation Guide:**
    - **Trend:** Upward trends indicate growth; downward trends suggest decline.
    - **Stability:** Lower CV indicates more consistent sales.
    - **Seasonality:** Peaks on certain days suggest customer preferences.
    - **Outliers:** Significant deviations may signal promotions or anomalies.
    """
)

st.markdown(
    """
    ---
    _This analysis helps identify stores with consistent performance, seasonal patterns, unusual spikes, and long-term trends. Use these insights for strategic planning and resource allocation._
    """
)


st.header("🧩 Store-Item Interaction")

st.subheader("Top Items per Store")

df_si = load_table(PreDataset.STORE_ITEM_WEEKLY)

store_id = st.selectbox(
    "Select store",
    sorted(df_si["store_nbr"].unique()),
)

top_n = st.slider(
    "Top N items", cfg.ui.top_n_min, cfg.ui.top_n_max, cfg.ui.top_n_default
)

df_store_items = df_si[df_si["store_nbr"] == store_id]

top_items = (
    df_store_items.groupby("item_nbr", as_index=False)
    .agg(total_sales=("unit_sales_sum", "sum"))
    .sort_values("total_sales", ascending=False)
    .head(top_n)
)


st.subheader("Weekly Seasonality by Item")

df_store_daily = load_table(PreDataset.STORE_ITEM_DAILY)
df_store_daily = df_store_daily[df_store_daily["store_nbr"] == store_id]

df_store_daily["dow"] = df_store_daily["date"].dt.day_name()

top_item_ids = top_items["item_nbr"].tolist()

seasonality_items = (
    df_store_daily[df_store_daily["item_nbr"].isin(top_item_ids)]
    .groupby(["item_nbr", "dow"], as_index=False)
    .agg(mean_sales=("unit_sales_sum", "mean"))
)

fig_item_season = px.line(
    seasonality_items,
    x="dow",
    y="mean_sales",
    color="item_nbr",
    title="Weekly Seasonality - Top Items",
)
render_plotly(fig_item_season)
