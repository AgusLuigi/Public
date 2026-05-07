"""
prophet_model.py

Interaktive Streamlit-Seite fuer Prophet-Modellierung.
Modularisierte Version mit robuster Pfad-Logik und zentralen Komponenten.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

# Logic for Folder
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Favorita_TSA.features.sarima_features import load_sarimax_segment
from Favorita_TSA.models.prophet_model import run_prophet_plotly
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

# UI Components
from streamlit_app.components.charts import render_plotly
from streamlit_app.components.metrics_row import render_metrics_row
from streamlit_app.components.filters import render_store_item_selector

# CONFIGURATION & DATAS
st.set_page_config(layout="wide", page_title="Prophet Model")
set_plotly_theme()
os.chdir(ROOT)

EXPERIMENT = getattr(cfg.mlflow, "prophet_experiment", "favorita_prophet_store_item")

# Feature-Gruppen
FEATURE_GROUPS: dict[str, list[str]] = {
    "Holidays": ["is_holiday_or_event", "pre_holiday", "post_holiday"],
    "Oil Price": ["oil_price", "oil_price_ma7", "oil_price_ma28"],
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

# UI-LOGIK
def render_prophet_params():
    """Rendert die Prophet-spezifischen Hyperparameter."""
    st.subheader("Prophet Hyperparameter")
    c1, c2 = st.columns(2)
    cp_scale = c1.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, format="%.3f")
    season_scale = c2.slider("Seasonality Prior Scale", 0.01, 10.0, 10.0)
    return cp_scale, season_scale

def render_exog_selector(is_daily: bool):
    """Rendert die Regressoren-Auswahl."""
    st.subheader("Exogene Regressoren")
    selected = []
    with st.expander("Regressoren wählen", expanded=False):
        for g_name, g_cols in FEATURE_GROUPS.items():
            st.markdown(f"**{g_name}**")
            cols = st.columns(min(len(g_cols), 4))
            for i, feat in enumerate(g_cols):
                with cols[i % len(cols)]:
                    if st.checkbox(feat, key=f"prophet_feat_{feat}"):
                        selected.append(feat)
    return selected

def render_history():
    """Stellt die MLflow Historie mit der originalen Rename-Map dar."""
    st.divider()
    st.subheader("MLflow Run History - Prophet")
    runs_df = _load_mlflow_runs()
    if runs_df.empty:
        st.info("Keine Runs gefunden.")
        return

    rename_map = {
        "tags.mlflow.runName": "Run Name",
        "params.pattern": "Pattern",
        "params.store": "Store",
        "params.item": "Item",
        "params.changepoint_prior_scale": "CP Scale",
        "params.seasonality_prior_scale": "Season Scale",
        "params.n_exog_features": "Regressoren #",
        "metrics.mae_primary": "MAE (Prophet)",
        "metrics.mae_naive": "MAE (Naive)",
        "metrics.r2_primary": "R2",
        "metrics.improvement_pct": "Improvement %",
        "start_time": "Zeitpunkt",
    }
    
    display_df = runs_df.rename(columns={c: v for c, v in rename_map.items() if c in runs_df.columns})
    
    # Formatierung (Logik-Erhalt)
    for col in ["MAE (Prophet)", "MAE (Naive)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if "R2" in display_df.columns:
        display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    if "Improvement %" in display_df.columns:
        display_df["Improvement %"] = display_df["Improvement %"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
        
    st.dataframe(display_df[[c for c in display_df.columns if c != "run_id"]].head(20), use_container_width=True)

# MAIN
def main():
    st.title("Prophet Model")
    st.caption("Facebook Prophet mit optionalen exogenen Regressoren.")

    # Filter
    pattern = st.selectbox("Demand Pattern", options=list(cfg.defaults.pattern_examples.__dict__.keys()))
    defaults = vars(getattr(cfg.defaults.pattern_examples, pattern))
    is_daily = "daily" in pattern
    
    store, item = render_store_item_selector(default_store=defaults["store"], default_item=defaults["item"])
    test_weeks = st.slider("Test Weeks", 1, 52, defaults["test_weeks"])

    st.divider()

    # Parameter & Regressoren
    cp_scale, season_scale = render_prophet_params()
    selected_features = render_exog_selector(is_daily)

    # Training
    if st.button("Run Prophet", type="primary", use_container_width=True):
        setup_mlflow(EXPERIMENT)
        with st.spinner("Prophet trainiert..."):
            try:
                df_seg = _load_segment(pattern)
                results, fig = run_prophet_plotly(
                    df=df_seg, store=store, item=item, pattern=pattern,
                    test_weeks=test_weeks, freq="D" if is_daily else "W",
                    changepoint_prior_scale=cp_scale,
                    seasonality_prior_scale=season_scale,
                    exog_cols=selected_features,
                    img_dir=IMG_DIR
                )
                st.session_state["prophet_res"] = results
                st.session_state["prophet_fig"] = fig
                _load_mlflow_runs.clear()
            except Exception as e:
                st.error(f"Fehler: {e}")

    # Ergebnisse
    if "prophet_res" in st.session_state:
        r = st.session_state["prophet_res"]
        st.divider()
        render_metrics_row(
            labels=["MAE Prophet", "MAE Naive", "R2", "Verbesserung", "Regressoren"],
            values=[f"{r['mae_primary']:.2f}", f"{r['mae_naive']:.2f}", f"{r['r2_primary']:.3f}",
                    f"{r['improvement_pct']:+.1f}%", str(r.get('n_exog_features', 0))]
        )
        render_plotly(st.session_state["prophet_fig"])

    # History
    render_history()

if __name__ == "__main__":
    main()