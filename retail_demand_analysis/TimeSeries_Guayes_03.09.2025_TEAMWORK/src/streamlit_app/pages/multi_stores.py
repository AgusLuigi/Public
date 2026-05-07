from __future__ import annotations

import plotly.express as px
import streamlit as st
import pandas as pd

from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.date_utils import normalize_time_col
from Favorita_TSA.utils.preprocess_data import load_table
from streamlit_app.components.charts import render_plotly

# DATEN-LOGIK
@st.cache_data(show_spinner="Lade Store-Daten...")
def load_store_data_all():
    """Lädt alle benötigten Granularitäten für die Store-Analyse."""
    daily = load_table(PreDataset.STORE_DAILY)
    weekly = load_table(PreDataset.STORE_WEEKLY)
    monthly = load_table(PreDataset.STORE_MONTHLY)
    for df, col in [(weekly, "week_start"), (weekly, "week")]:
        if col in weekly.columns:
            weekly = normalize_time_col(weekly, col)
            weekly["week_ts"] = weekly[col]
            break
            
    return daily, weekly, monthly

def calculate_stability(df_weekly: pd.DataFrame, store_ids: list[int]):
    """Berechnet CV (Coefficient of Variation) für die ausgewählten Stores."""
    df_sel = df_weekly[df_weekly["store_nbr"].isin(store_ids)]
    stability = df_sel.groupby("store_nbr", as_index=False).agg(
        mean_sales=("unit_sales_sum", "mean"),
        std_sales=("unit_sales_sum", "std"),
    )
    stability["cv"] = stability["std_sales"] / stability["mean_sales"]
    return stability.sort_values("cv")

def calculate_outliers(df_weekly: pd.DataFrame, store_ids: list[int]):
    """Berechnet Z-Scores basierend auf Rolling Mean/Std."""
    df_out = df_weekly[df_weekly["store_nbr"].isin(store_ids)].copy().sort_values("week")
    _rw = cfg.analysis.rolling_window
    _rp = cfg.analysis.rolling_min_periods
    _z_thresh = cfg.analysis.zscore_threshold
    
    df_out["rolling_mean"] = df_out.groupby("store_nbr")["unit_sales_sum"].transform(
        lambda s: s.rolling(_rw, min_periods=_rp).mean()
    )
    df_out["rolling_std"] = df_out.groupby("store_nbr")["unit_sales_sum"].transform(
        lambda s: s.rolling(_rw, min_periods=_rp).std()
    )
    df_out["z_score"] = (df_out["unit_sales_sum"] - df_out["rolling_mean"]) / df_out["rolling_std"]
    df_out["is_outlier"] = df_out["z_score"].abs() > _z_thresh
    return df_out, _z_thresh

# UI-KOMPONENTEN
def render_store_selector(all_stores: list[int]):
    """Zentrale Store-Auswahl mit Validierung."""
    store_ids = st.multiselect(
        "Select stores",
        options=sorted(all_stores),
        default=[cfg.ui.default_store],
    )
    if not store_ids:
        st.info("Please select at least one store.")
        st.stop()
    if len(store_ids) > 10:
        st.warning("Please select at most 10 stores for readability.")
    return store_ids

def render_trend_section(df_daily, df_weekly, df_monthly, store_ids):
    st.header("Trend Analysis")
    st.caption("Long-term sales trends (Monthly, Weekly, Daily).")
    tabs = st.tabs(["Monthly", "Weekly", "Daily"])
    with tabs[0]:
        fig = px.line(df_monthly[df_monthly["store_nbr"].isin(store_ids)], 
                     x="month", y="unit_sales_sum", color="store_nbr", title="Monthly Sales")
        render_plotly(fig)
    with tabs[1]:
        fig = px.line(df_weekly[df_weekly["store_nbr"].isin(store_ids)], 
                     x="week", y="unit_sales_sum", color="store_nbr", title="Weekly Sales")
        render_plotly(fig)
    with tabs[2]:
        fig = px.line(df_daily[df_daily["store_nbr"].isin(store_ids)], 
                     x="date", y="unit_sales_sum", color="store_nbr", title="Daily Sales")
        render_plotly(fig)

# MAIN
def main():
    st.title("Store Analysis")
    st.caption("Vergleichende Analyse von Trends, Stabilität und Saisonalität auf Store-Ebene.")
    # 1. Daten laden
    df_daily, df_weekly, df_monthly = load_store_data_all()
    # 2. Filter
    store_ids = render_store_selector(df_daily["store_nbr"].unique())
    st.divider()
    # 3. Trend Sektion
    render_trend_section(df_daily, df_weekly, df_monthly, store_ids)
    st.divider()
    # 4. Stabilität
    st.header("Store Stability")
    st.caption("Niedrigere Variabilität (CV) bedeutet stabilere Verkäufe.")
    stability_df = calculate_stability(df_weekly, store_ids)
    
    c1, c2 = st.columns([1, 2])
    c1.dataframe(stability_df[["store_nbr", "cv"]], use_container_width=True, hide_index=True)
    with c2:
        fig_stab = px.bar(stability_df, x="store_nbr", y="cv", color="store_nbr", 
                         title="Sales Variability (CV)")
        render_plotly(fig_stab)
    st.divider()
    # 5. Saisonalität
    st.header("Seasonality")
    df_daily["dow"] = df_daily["date"].dt.day_name()
    seasonality = df_daily[df_daily["store_nbr"].isin(store_ids)].groupby(["store_nbr", "dow"], as_index=False).agg(
        mean_sales=("unit_sales_sum", "mean")
    )
    fig_seas = px.bar(seasonality, x="dow", y="mean_sales", color="store_nbr", barmode="group",
                     category_orders={"dow": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]},
                     title="Average Sales by Day of Week")
    render_plotly(fig_seas)
    st.divider()
    # 6. Outliers
    st.header("Outliers")
    df_out, z_thresh = calculate_outliers(df_weekly, store_ids)
    fig_out = px.scatter(df_out, x="week", y="unit_sales_sum", color="is_outlier",
                        hover_data=["z_score", "store_nbr"],
                        title=f"Weekly Sales with Outliers (|Z| > {z_thresh})")
    render_plotly(fig_out)
    # 7. Store-Item Interaction
    with st.expander("Store-Item Interaction"):
        df_si = load_table(PreDataset.STORE_ITEM_WEEKLY)
        store_id = st.selectbox("Select single store for item analysis", sorted(df_si["store_nbr"].unique()))
        st.write(f"Detailanalyse für Store {store_id}...")

if __name__ == "__main__":
    main()