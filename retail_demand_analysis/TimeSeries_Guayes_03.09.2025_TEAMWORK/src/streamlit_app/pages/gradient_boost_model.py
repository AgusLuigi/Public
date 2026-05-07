"""
gradient_boost_model.py

Interaktive Streamlit-Seite fuer GradientBoosting-Modellierung.
Nutzt zentrale UI-Komponenten für Filter, Metriken und Charts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

# 1. Pfad-Logik & Imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Favorita_TSA.features.sarima_features import load_sarimax_segment
from Favorita_TSA.models.gradient_boost import run_gradient_boost_plotly
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

# Zentrale UI-Komponenten
from streamlit_app.components.charts import render_plotly
from streamlit_app.components.metrics_row import render_metrics_row
from streamlit_app.components.filters import render_store_item_selector

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION & DATEN-LOGIK
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Gradient Boosting Model")
set_plotly_theme()
os.chdir(ROOT)

EXPERIMENT = getattr(cfg.mlflow, "gb_experiment", "favorita_gb_store_item")

FEATURE_GROUPS: dict[str, list[str]] = {
    "Holidays": ["is_holiday_or_event", "pre_holiday", "post_holiday"],
    "Oil Price": ["oil_price", "oil_price_ma7", "oil_price_ma28", "oil_price_pct_change"],
    "Calendar": ["is_weekend", "is_payday", "is_month_start", "is_month_end", "days_to_next_holiday", "days_since_last_holiday"],
    "Transactions": ["transactions", "transactions_ma7", "transactions_z_score"],
    "Promotion": ["onpromotion", "promo_streak", "promo_rate_7d"],
    "Store / Item": ["store_cluster", "perishable", "store_type", "family"],
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

# ─────────────────────────────────────────────────────────────────────────────
# UI-LOGIK (ERHALT DER VOLLSTÄNDIGKEIT)
# ─────────────────────────────────────────────────────────────────────────────

def render_exogenous_features(is_daily: bool):
    """Erhält die vollständige Feature-Gruppen-Logik inkl. 'Alle'-Checkboxen."""
    st.subheader("Exogene Features")
    selected_features = []
    
    with st.expander("Feature-Gruppen auswaehlen", expanded=True):
        for group_name, group_cols in FEATURE_GROUPS.items():
            display_cols = group_cols if is_daily else [c for c in group_cols if c != "is_weekend"]
            if not display_cols: continue

            g_col1, g_col2 = st.columns([1, 5])
            all_key = f"gb_all_{group_name}"
            
            # Callback-Logik für "Alle auswählen"
            select_all = g_col1.checkbox("Alle", key=all_key, help=f"Gruppe '{group_name}'")
            
            g_col2.markdown(f"**{group_name}**")
            feat_cols_ui = g_col2.columns(min(len(display_cols), 4))
            for i, feat in enumerate(display_cols):
                with feat_cols_ui[i % len(feat_cols_ui)]:
                    # Falls "Alle" gewählt wurde, wird der Default-Wert überschrieben
                    val = st.checkbox(feat, key=f"gb_feat_{group_name}_{feat}", value=select_all)
                    if val: selected_features.append(feat)
    return selected_features

def render_mlflow_history():
    """Vollständige Integration deiner rename_map und Formatierung."""
    st.subheader("MLflow Run History - GradientBoosting")
    if st.button("Aktualisieren"):
        _load_mlflow_runs.clear()
        
    runs_df = _load_mlflow_runs()
    if runs_df.empty:
        st.info("Noch keine Runs gefunden.")
        return

    rename_map = {
        "tags.mlflow.runName": "Run Name", "params.pattern": "Pattern",
        "params.store": "Store", "params.item": "Item", "params.freq": "Freq",
        "params.n_estimators": "n_est", "params.max_depth": "depth",
        "params.learning_rate": "lr", "params.lag_periods": "Lags",
        "params.n_features": "Features #", "metrics.mae_primary": "MAE (GBM)",
        "metrics.mae_naive": "MAE (Naive)", "metrics.r2_primary": "R2",
        "metrics.improvement_pct": "Improvement %", "start_time": "Zeitpunkt",
    }
    
    display_df = runs_df.rename(columns={c: v for c, v in rename_map.items() if c in runs_df.columns})
    
    # Formatierung (Logik-Erhalt)
    for col in ["MAE (GBM)", "MAE (Naive)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if "R2" in display_df.columns:
        display_df["R2"] = display_df["R2"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
    if "Improvement %" in display_df.columns:
        display_df["Improvement %"] = display_df["Improvement %"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
    
    st.dataframe(display_df[[c for c in display_df.columns if c != "run_id"]].head(30), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("Gradient Boosting Model")
    st.caption("GBM mit Lag-Features und exogenen Daten.")

    # 1. Daten- & Store-Auswahl
    pattern = st.selectbox("Demand Pattern", options=list(cfg.defaults.pattern_examples.__dict__.keys()))
    defaults = vars(getattr(cfg.defaults.pattern_examples, pattern))
    is_daily = "daily" in pattern
    
    store, item = render_store_item_selector(default_store=defaults["store"], default_item=defaults["item"])
    
    c_test, c_trim = st.columns(2)
    test_weeks = c_test.slider("Test Weeks", 1, 52, defaults["test_weeks"])
    trim_days = c_trim.slider("Trim Zero-Phase", 0, 90, 0, step=5, disabled=not is_daily)

    st.divider()

    # 2. Lag-Features
    st.subheader("Lag-Features")
    all_lags = [1, 2, 3, 7, 14, 21, 28, 56] if is_daily else [1, 2, 4, 8, 13, 26, 52]
    lag_periods = st.multiselect("Lag-Perioden", options=all_lags, default=all_lags[:4])

    # 3. Exogene Features (Erhalt der Checkbox-Logik)
    selected_features = render_exogenous_features(is_daily)

    # 4. GBM Hyperparameter
    st.divider()
    h1, h2, h3, h4 = st.columns(4)
    n_est = h1.slider("n_estimators", 50, 500, 200, 50)
    depth = h2.slider("max_depth", 2, 8, 4)
    lr = h3.slider("learning_rate", 0.01, 0.3, 0.05)
    s_val = h4.slider("s (Saison-Benchmark)", 1, 52, defaults["season"])

    # 5. Training
    if st.button("Run GradientBoosting", type="primary", use_container_width=True):
        setup_mlflow(EXPERIMENT)
        with st.spinner("Training..."):
            try:
                df_seg = _load_segment(pattern)
                results, fig = run_gradient_boost_plotly(
                    df=df_seg, pattern=pattern, store=int(store), item=int(item),
                    freq="D" if is_daily else "W", season_length=int(s_val),
                    test_weeks=int(test_weeks), feature_cols=selected_features,
                    lag_periods=sorted(lag_periods), n_estimators=int(n_est),
                    max_depth=int(depth), learning_rate=float(lr),
                    trailing_zero_min_days=int(trim_days) if is_daily else 0,
                    img_dir=IMG_DIR
                )
                st.session_state["gb_res"] = results
                st.session_state["gb_fig"] = fig
                _load_mlflow_runs.clear()
            except Exception as e:
                st.error(f"Fehler: {e}")

    # 6. Ergebnisse
    if "gb_res" in st.session_state:
        r = st.session_state["gb_res"]
        st.divider()
        render_metrics_row(
            labels=["MAE GBM", "MAE Naive", "R2 GBM", "Verbesserung", "Train/Test", "Features"],
            values=[f"{r['mae_primary']:.2f}", f"{r['mae_naive']:.2f}", f"{r['r2_primary']:.3f}",
                    f"{r['improvement_pct']:+.1f}%", f"{r['train_size']}/{r['test_size']}", str(r['n_features'])]
        )
        render_plotly(st.session_state["gb_fig"])

    # 7. Historie
    st.divider()
    render_mlflow_history()

if __name__ == "__main__":
    main()