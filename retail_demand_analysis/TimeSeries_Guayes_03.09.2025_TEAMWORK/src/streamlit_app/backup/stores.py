import plotly.express as px
import streamlit as st

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

st.title("Store Analysis")

df_store_daily = load_table(PreDataset.STORE_DAILY)
df_store_weekly = load_table(PreDataset.STORE_WEEKLY)
df_store_monthly = load_table(PreDataset.STORE_MONTHLY)

# Make timeseries columns for plotting
df_store_weekly["week_ts"] = df_store_weekly["week"].dt.start_time
df_store_monthly["month_ts"] = df_store_monthly["month"].dt.to_timestamp()


store_id = st.selectbox(
    "Select store",
    sorted(df_store_daily["store_nbr"].unique()),
)

df_store_daily_choice = df_store_daily[df_store_daily["store_nbr"] == store_id]
df_store_weekly_choice = df_store_weekly[df_store_weekly["store_nbr"] == store_id]
df_store_monthly_choice = df_store_monthly[df_store_monthly["store_nbr"] == store_id]


fig_daily = px.line(
    df_store_daily_choice,
    x="date",
    y="unit_sales_sum",
    color="store_nbr",
    title=f"Store {store_id} - Daily Sales",
)
st.plotly_chart(fig_daily, use_container_width=True)


fig_weekly = px.line(
    df_store_weekly_choice,
    x="week_ts",
    y="unit_sales_sum",
    color="store_nbr",
    title=f"Store {store_id} - Weekly Sales",
)
st.plotly_chart(fig_weekly, use_container_width=True)


fig_monthly = px.line(
    df_store_monthly_choice,
    x="month_ts",
    y="unit_sales_sum",
    color="store_nbr",
    title=f"Store {store_id} - Monthly Sales",
)
st.plotly_chart(fig_monthly, use_container_width=True)
