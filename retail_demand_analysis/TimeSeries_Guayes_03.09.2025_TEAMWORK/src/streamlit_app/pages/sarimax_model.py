"""
sarimax_model.py

Interactive Streamlit page for SARIMAX modeling.
Modular version with robust path logic and central components.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Favorita_TSA.features.sarima_features import load_sarimax_segment
from Favorita_TSA.models.sarimax import (
    run_sarimax_feature_search,
    run_sarimax_grid_search,
    run_sarimax_plotly,
)
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

from streamlit_app.components.charts import render_plotly
from streamlit_app.components.metrics_row import render_metrics_row
from streamlit_app.components.filters import render_store_item_selector

st.set_page_config(layout="wide", page_title="SARIMAX Model")
set_plotly_theme()
os.chdir(ROOT)

EXPERIMENT = cfg.mlflow.sarimax_experiment

FEATURE_GROUPS: dict[str, list[str]] = {
    "Holidays": ["is_holiday_or_event", "pre_holiday", "post_holiday"],
    "Oil Price": ["oil_price", "oil_price_ma7", "oil_price_ma28"],
    "Calendar": ["is_payday", "is_month_start", "is_month_end"],
    "Transactions": ["transactions", "transactions_ma7"],
    "Promotion": ["onpromotion", "promo_streak"],
}

@st.cache_data(show_spinner="Lade Segment-Daten...")
def _load_segment(pattern: str) -> pd.DataFrame:
    return load_sarimax_segment(pattern)

@st.cache_data(ttl=15, show_spinner=False)
def _load_mlflow_runs() -> pd.DataFrame:
    setup_mlflow(EXPERIMENT)
    try:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
        return runs if not runs.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# UI COMPONENTS (LOGIC PRESERVED)
def render_sarima_params(is_daily: bool, default_s: int):
    """Renders the sliders for (p,d,q) and (P,D,Q,s)."""
    st.subheader("SARIMA Parameters")
    c1, c2, c3 = st.columns(3)
    p = c1.slider("p (AR)", 0, 5, 1)
    d = c2.slider("d (Diff)", 0, 2, 1)
    q = c3.slider("q (MA)", 0, 5, 1)

    st.caption("Seasonal Components")
    c4, c5, c6, c7 = st.columns(4)
    P = c4.slider("P (S-AR)", 0, 2, 0)
    D = c5.slider("D (S-Diff)", 0, 1, 0)
    Q = c6.slider("Q (S-MA)", 0, 2, 0)
    s = c7.slider("s (Season)", 1, 52, default_s)
    
    return (p, d, q), (P, D, Q, s)

def render_exog_selector():
    """Renders the selection of exogenous features."""
    st.subheader("Exogenous Features")
    selected = []
    with st.expander("Select Features", expanded=False):
        for g_name, g_cols in FEATURE_GROUPS.items():
            st.markdown(f"**{g_name}**")
            cols = st.columns(min(len(g_cols), 4))
            for i, feat in enumerate(g_cols):
                if cols[i % len(cols)].checkbox(feat, key=f"sarimax_feat_{feat}"):
                    selected.append(feat)
    return selected

def render_history():
    """Complete rename_map Integration."""
    st.divider()
    st.subheader("MLflow Run History - SARIMAX")
    runs_df = _load_mlflow_runs()
    if runs_df.empty:
        st.info("No runs found.")
        return

    rename_map = {
        "tags.mlflow.runName": "Run Name",
        "params.pattern": "Pattern",
        "params.store": "Store", 
        "params.item": "Item", 
        "params.freq": "Freq",
        "params.p": "p", 
        "params.d": "d", 
        "params.q": "q",
        "params.s_p": "P", 
        "params.s_d": "D", 
        "params.s_q": "Q", 
        "params.s": "s",
        "params.n_exog_features": "Exog #",
        "metrics.mae_primary": "MAE (SARIMAX)", 
        "metrics.mae_naive": "MAE (Naive)",
        "metrics.r2_primary": "R2", 
        "metrics.improvement_pct": "Improvement %",
        "start_time": "Time",
    }
    
    display_df = runs_df.rename(columns={c: v for c, v in rename_map.items() if c in runs_df.columns})
    
    # Formatting (identical to original)
    for col in ["MAE (SARIMAX)", "MAE (Naive)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if "R2" in display_df.columns:
        display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    if "Improvement %" in display_df.columns:
        display_df["Improvement %"] = display_df["Improvement %"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
        
    st.dataframe(display_df[[c for c in display_df.columns if c != "run_id"]].head(20), use_container_width=True)

# MAIN
def main():
    st.title("SARIMAX Model")
    st.caption("Classic Time Series Analysis with Exogenous Regressors.")

    # 1. Basic Filter
    pattern = st.selectbox("Demand Pattern", options=list(cfg.defaults.pattern_examples.__dict__.keys()))
    defaults = vars(getattr(cfg.defaults.pattern_examples, pattern))
    is_daily = "daily" in pattern
    
    store, item = render_store_item_selector(default_store=defaults["store"], default_item=defaults["item"])
    test_weeks = st.slider("Test Weeks", 1, 52, defaults["test_weeks"])

    st.divider()

    # 2. Parameter & Features
    (p, d, q), (P, D, Q, s) = render_sarima_params(is_daily, defaults["season"])
    selected_features = render_exog_selector()

    # 3. Actions (Buttons)
    st.divider()
    b1, b2, b3 = st.columns(3)
    
    if b1.button("Run SARIMAX", type="primary", use_container_width=True):
        setup_mlflow(EXPERIMENT)
        with st.spinner("Training..."):
            try:
                df_seg = _load_segment(pattern)
                results, fig = run_sarimax_plotly(
                    df=df_seg, store=store, item=item, pattern=pattern,
                    test_weeks=test_weeks, freq="D" if is_daily else "W",
                    order=(p, d, q), seasonal_order=(P, D, Q, s),
                    exog_cols=selected_features, img_dir=IMG_DIR
                )
                st.session_state["sx_res"] = results
                st.session_state["sx_fig"] = fig
                _load_mlflow_runs.clear()
            except Exception as e:
                st.error(f"Fehler: {e}")

    if b2.button("Grid Search (p,d,q)", use_container_width=True):
        st.info("Grid Search started (see MLflow)...")
        df_seg = _load_segment(pattern)
        run_sarimax_grid_search(df_seg, store, item, pattern, test_weeks, "D" if is_daily else "W", s)

    if b3.button("Feature Search", use_container_width=True):
        st.info("Feature Search started...")
        df_seg = _load_segment(pattern)
        run_sarimax_feature_search(df_seg, store, item, pattern, test_weeks, "D" if is_daily else "W", (p,d,q), (P,D,Q,s))

    # 4. Results
    if "sx_res" in st.session_state:
        r = st.session_state["sx_res"]
        st.divider()
        render_metrics_row(
            labels=["MAE SARIMAX", "MAE Naive", "R2", "Improvement", "Parameter"],
            values=[f"{r['mae_primary']:.2f}", f"{r['mae_naive']:.2f}", f"{r['r2_primary']:.3f}",
                    f"{r['improvement_pct']:+.1f}%", f"({p},{d},{q})({P},{D},{Q},{s})"]
        )
        render_plotly(st.session_state["sx_fig"])

    # 5. History
    render_history()

if __name__ == "__main__":
    main()