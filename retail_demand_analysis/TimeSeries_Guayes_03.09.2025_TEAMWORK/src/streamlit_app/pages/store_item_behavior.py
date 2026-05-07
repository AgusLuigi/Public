"""
store_item_behavior.py

Analysis of store-item behavior (decomposition, seasonality, outlier).
Modularized version with Session-State caching and hybrid dynamic filters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

# ROOT PATH
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# IMPORTS
from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.paths import PREPROCESSED_DIR
from Favorita_TSA.utils.preprocess_data import load_table
from Favorita_TSA.viz.color_manager import ColorManager, apply_modern_theme
from streamlit_app.components.charts import render_plotly
from streamlit_app.components.metrics_row import render_metrics_row

# HIGH-SPEED DATA PERSISTENCE (Session State)
def get_persist_data():
    """Hält die Daten im RAM des Browsers/Servers für sofortiges Umschalten."""
    if "df_daily" not in st.session_state:
        with st.spinner("Initiales Laden der Verkaufsdaten (einmalig)..."):
            st.session_state["df_daily"] = load_table(PreDataset.STORE_ITEM_DAILY)
            st.session_state["df_weekly"] = load_table(PreDataset.STORE_ITEM_WEEKLY)
    return st.session_state["df_daily"], st.session_state["df_weekly"]

@st.cache_data(show_spinner="Lade Feiertags-Kontext...")
def load_holiday_data_lazy(is_daily: bool):
    fname = "store_item_daily_holiday.parquet" if is_daily else "store_item_weekly_holiday.parquet"
    p = PREPROCESSED_DIR / fname
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

# ANALYSIS & PLOTTING
def perform_stl_decomposition(series: pd.Series, period: int):
    if len(series) < period * 2:
        return None, None, None
    stl = STL(series, period=period, robust=True).fit()
    return stl.trend, stl.seasonal, stl.resid

def plot_behavior_decomposition(ts_dense, hol_dense, trend, seasonal, resid, title):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Observed (Dense)", "Trend", "Seasonal", "Residuals")
    )
    
    colors_ns = ColorManager.get_colors()
    main_color = getattr(colors_ns, "primary_color", "#0068c9") 
    
    if not hol_dense.empty and "holiday_event" in hol_dense.columns:
        custom_data = np.stack([
            hol_dense["holiday_event"].fillna("None"),
            hol_dense["pre_holiday"].fillna(0),
            hol_dense["post_holiday"].fillna(0)
        ], axis=-1)
        hovertemplate = "Date: %{x}<br>Value: %{y:.2f}<br>Event: %{customdata[0]}<br>Pre: %{customdata[1]}d<br>Post: %{customdata[2]}d"
    else:
        custom_data, hovertemplate = None, "Date: %{x}<br>Value: %{y:.2f}"

    plots = [ts_dense, trend, seasonal, resid]
    for i, data in enumerate(plots, 1):
        fig.add_trace(go.Scatter(
            x=data.index, y=data, mode="lines", line=dict(color=main_color, width=1.5),
            customdata=custom_data, hovertemplate=hovertemplate, name=f"Row {i}"
        ), row=i, col=1)

    apply_modern_theme(fig)
    fig.update_layout(height=900, title_text=title, showlegend=False)
    return fig

# MAIN
def main():
    st.set_page_config(layout="wide", page_title="Behavior Analysis")
    os.chdir(ROOT)

    # Banner Recovery
    banner_path = ROOT / "assets" / "behavior_analysis_banner.png" 
    if banner_path.exists():
        st.image(str(banner_path), use_container_width=True)

    # 1. Schnelles Laden aus Session State
    df_daily, df_weekly = get_persist_data()

    st.title("Store-Item Behavior Analysis")
    st.caption("Echtzeit-Analyse durch persistentes Caching.")

    # 2. Hybride Filter-Logik
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        all_stores = sorted(df_daily["store_nbr"].unique())
        store = st.selectbox("Store Nr.", options=all_stores, index=0)

    with c2:
        granularity = st.radio("Granularity", ["Daily", "Weekly"], horizontal=True)
        is_daily = granularity == "Daily"
        df_base = df_daily if is_daily else df_weekly
        
        # Dynamische Item-Liste für diesen Store
        available_items = sorted(df_base[df_base["store_nbr"] == store]["item_nbr"].unique())
        
        # selectbox erlaubt Tippen UND Auswählen
        item = st.selectbox(
            "Item Nr. (Tippen & Enter oder Wählen)",
            options=available_items,
            help=f"Dieser Store führt {len(available_items)} Items."
        )

    with c3:
        show_sparse = st.checkbox("Show sparse series", value=False)
        if st.button("♻️ Cache zurücksetzen"):
            for k in ["df_daily", "df_weekly"]: 
                if k in st.session_state: del st.session_state[k]
            st.cache_data.clear()
            st.rerun()

    # 3. Daten-Vorbereitung (Subset ist nun garantiert nicht leer)
    mask = (df_base["store_nbr"] == store) & (df_base["item_nbr"] == item)
    subset = df_base[mask].sort_values("date" if is_daily else "week")
    
    # 4. Holiday Context & Time Series
    df_hol = load_holiday_data_lazy(is_daily)
    season_lag = 7 if is_daily else 52
    time_col = "date" if is_daily else "week"
    subset[time_col] = pd.to_datetime(subset[time_col])
    ts = subset.set_index(time_col)["unit_sales_sum"]
    
    full_idx = pd.date_range(ts.index.min(), ts.index.max(), freq="D" if is_daily else "W-MON")
    ts_dense = ts.reindex(full_idx, fill_value=0)
    
    hol_dense = pd.DataFrame(index=full_idx)
    if not df_hol.empty:
        mask_h = (df_hol["store_nbr"] == store) & (df_hol["item_nbr"] == item)
        hol_subset = df_hol[mask_h].sort_values(time_col).set_index(time_col)
        if not hol_subset.empty:
            hol_dense = hol_subset.reindex(full_idx)

    # 5. Analysis & Metrics
    trend, seasonal, resid = perform_stl_decomposition(ts_dense, season_lag)
    
    if trend is None:
        st.error("Nicht genügend Datenpunkte für die Dekomposition.")
        st.stop()

    cv = ts_dense.std() / ts_dense.mean() if ts_dense.mean() != 0 else 0
    acf_val = acf(ts_dense, nlags=season_lag)[-1]
    zero_share = (ts_dense == 0).mean()

    st.divider()
    render_metrics_row(
        labels=["Coefficient of variation (CV)", f"Seasonal ACF (@{season_lag})", "Zero Share (%)"],
        values=[f"{cv:.2f}", f"{acf_val:.3f}", f"{zero_share:.1%}"]
    )

    # 6. Plotting
    st.divider()
    fig = plot_behavior_decomposition(
        ts_dense, hol_dense, trend, seasonal, resid,
        title=f"{granularity} Decomposition · Store {store} · Item {item}"
    )
    render_plotly(fig)

    if show_sparse:
        st.subheader("🧾 Sparse series (sales periods only)")
        ts_sparse = ts[ts > 0]
        if not ts_sparse.empty:
            fig_sparse = go.Figure(go.Scatter(x=ts_sparse.index, y=ts_sparse.values, mode="lines+markers"))
            fig_sparse.update_layout(height=300, title="Sparse series (no zero-filled periods)")
            st.plotly_chart(fig_sparse, use_container_width=True)

    with st.expander("Interpretation Guide"):
        st.markdown("""
        **Seasonality:** Ein hoher ACF-Wert deutet auf starke regelmäßige Muster hin.
        **Trend:** Zeigt die langfristige Entwicklung ohne Rauschen.
        **Residuals:** Nicht erklärte Abweichungen. Spikes hier deuten oft auf Sonderaktionen oder Datenfehler hin.
        """)

if __name__ == "__main__":
    main()