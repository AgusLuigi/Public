"""
baseline_model.py

Interaktive Streamlit-Seite fuer Baseline-Modelltraining.
Modularisierte Version zur einfachen Fehlerbehebung ohne Code-Schwund.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import streamlit as st

# Pfad-Logik für Unterordner-Struktur
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Favorita_TSA.models.baseline import (
    aggregate_to_weekly,
    run_baseline_plotly,
    run_grid_search_cv,
)
from Favorita_TSA.models.data_preparation import build_dataframes
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

# UI-Komponenten aus dem Projekt
from streamlit_app.components.charts import render_plotly
from streamlit_app.components.metrics_row import render_metrics_row

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION & GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Baseline Model")
set_plotly_theme()

# Sicherstellen, dass wir im Projekt-Root arbeiten
os.chdir(ROOT)

EXPERIMENT = cfg.mlflow.experiment

PATTERN_KEY_MAP = {
    "daily_smooth": "smooth_daily",
    "daily_erratic": "erratic_daily",
    "weekly_smooth": "smooth_weekly",
    "weekly_erratic": "erratic_weekly",
}

_pe = cfg.defaults.pattern_examples
PATTERN_DEFAULTS: dict[str, dict] = {
    "daily_smooth": vars(_pe.daily_smooth),
    "daily_erratic": vars(_pe.daily_erratic),
    "weekly_smooth": vars(_pe.weekly_smooth),
    "weekly_erratic": vars(_pe.weekly_erratic),
}

# ─────────────────────────────────────────────────────────────────────────────
# DATEN-FUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Lade Daten (einmalig) ...")
def _load_all_dataframes() -> dict[str, pd.DataFrame]:
    return build_dataframes()

def get_dataframe(pattern: str) -> pd.DataFrame:
    return _load_all_dataframes()[PATTERN_KEY_MAP[pattern]]

@st.cache_data(ttl=15, show_spinner=False)
def load_mlflow_runs() -> pd.DataFrame:
    setup_mlflow(EXPERIMENT)
    try:
        runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
        if runs.empty: return pd.DataFrame()
        
        wanted = [
            "run_id", "tags.mlflow.runName", "params.pattern", "params.model_type",
            "params.store", "params.item", "params.freq", "params.season_length",
            "params.test_weeks", "params.mp_max_p", "params.mp_s_max_p", "params.mp_max_q",
            "params.mp_s_max_q", "params.mp_d", "params.mp_seasonal",
            "params.mp_decomposition_type", "metrics.mae_primary", "metrics.mae_naive",
            "metrics.improvement_pct", "start_time"
        ]
        existing = [c for c in wanted if c in runs.columns]
        return runs[existing].head(30)
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# UI-BLOCKS (CHIRURGISCHE TRENNUNG)
# ─────────────────────────────────────────────────────────────────────────────

def render_data_selection(defaults, is_weekly):
    """Rendert die Eingabefelder für die Datenauswahl."""
    st.subheader("Daten")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    store = c2.number_input("Store Nr.", 1, 54, defaults["store"])
    item = c3.number_input("Item Nr.", 1, None, defaults["item"])
    
    gap_pct = c4.slider("Gap Threshold", 1, 20, 5, format="%d%%")
    gap_threshold = gap_pct / 100
    
    trim_days = c5.slider("Trim Zero-Phase", 0, 90, cfg.models.trailing_zero_min_days, 
                          step=5, format="%d Tage", disabled=is_weekly)
    trailing_zeros = int(trim_days) if not is_weekly else 0
    
    return store, item, gap_threshold, trailing_zeros

def render_model_basics(defaults):
    """Rendert die Basis-Modell-Einstellungen."""
    st.subheader("Modell")
    c1, c2, c3 = st.columns(3)
    
    m_type = c1.radio("Modelltyp", ["sarima", "theta"], 
                      index=0 if defaults["model"] == "sarima" else 1, 
                      format_func=str.upper, horizontal=True)
    season = c2.slider("Season Length", 1, 52, defaults["season"])
    test_w = c3.slider("Test Weeks", 1, 52, defaults["test_weeks"])
    
    return m_type, season, test_w

def render_sarima_parameters():
    """Rendert das vollständige Parameter-Set für AutoARIMA."""
    st.subheader("AutoARIMA Parameter")
    _a = cfg.models.autoarima
    
    # Max-Grenzen
    c = st.columns(6)
    max_p = c[0].number_input("max_p", 0, 10, _a.max_p)
    max_q = c[1].number_input("max_q", 0, 10, _a.max_q)
    max_d = c[2].number_input("max_d", 0, 3, _a.max_d)
    max_P = c[3].number_input("max_P", 0, 5, _a.max_P)
    max_Q = c[4].number_input("max_Q", 0, 5, _a.max_Q)
    max_D = c[5].number_input("max_D", 0, 2, _a.max_D)

    # Start-Werte
    s = st.columns(6)
    start_p = s[0].number_input("start_p", 0, 10, _a.start_p)
    start_q = s[1].number_input("start_q", 0, 10, _a.start_q)
    start_P = s[2].number_input("start_P", 0, 5, _a.start_P)
    start_Q = s[3].number_input("start_Q", 0, 5, _a.start_Q)

    # Automatik & Seasonal
    a = st.columns(4)
    d_auto = a[0].checkbox("d = auto", value=True)
    D_auto = a[1].checkbox("D = auto", value=True)
    seasonal = a[2].checkbox("Seasonal", value=True)

    params = {
        "max_p": int(max_p), "max_q": int(max_q), "max_d": int(max_d),
        "max_P": int(max_P), "max_Q": int(max_Q), "max_D": int(max_D),
        "start_p": int(start_p), "start_q": int(start_q),
        "start_P": int(start_P), "start_Q": int(start_Q),
        "seasonal": seasonal,
    }
    if not d_auto: params["d"] = int(a[0].number_input("d (fix)", 0, 3, 0))
    if not D_auto: params["D"] = int(a[1].number_input("D (fix)", 0, 2, 0))
    
    return params

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.title("Baseline Model")
    st.caption("Konfiguriere und trainiere Baseline-Modelle — alle Runs werden in MLflow gespeichert.")
    st.divider()

    # 1. Auswahl Pattern
    pattern = st.selectbox("Demand Pattern", options=list(PATTERN_DEFAULTS.keys()))
    defaults = PATTERN_DEFAULTS[pattern]
    is_daily = "daily" in pattern

    # 2. UI-Komponenten rendern
    store, item, gap_threshold, trailing_zeros = render_data_selection(defaults, not is_daily)
    st.divider()
    model_type, season_length, test_weeks = render_model_basics(defaults)
    st.divider()

    # 3. Modell Parameter
    if model_type == "sarima":
        model_params = render_sarima_parameters()
    else:
        st.subheader("Theta Parameter")
        decomp = st.radio("Decomposition Type", ["multiplicative", "additive"], horizontal=True)
        model_params = {"decomposition_type": decomp}

    st.divider()

    # 4. Training Trigger
    if st.button("Run Model", use_container_width=True, type="primary"):
        execute_training(pattern, store, item, season_length, test_weeks, 
                         model_type, gap_threshold, trailing_zeros, model_params, is_daily)

    # 5. Ergebnisse & Grid Search & History
    display_results()
    render_grid_search_cv_section(pattern, store, item, season_length, is_daily, gap_threshold, trailing_zeros)
    render_history()

def execute_training(pattern, store, item, season, test_w, m_type, gap, trim, params, is_daily):
    setup_mlflow(EXPERIMENT)
    with st.spinner(f"Trainiere {m_type.upper()}..."):
        try:
            df = get_dataframe(pattern)
            if not is_daily:
                df = aggregate_to_weekly(df, store=int(store), item=int(item))
            
            results, fig = run_baseline_plotly(
                df=df, pattern=pattern, store=int(store), item=int(item),
                freq="D" if is_daily else "W", season_length=int(season),
                test_weeks=int(test_w), model_type=m_type,
                gap_threshold=float(gap), trailing_zero_min_days=trim,
                model_params=params, img_dir=IMG_DIR
            )
            st.session_state["last_results"] = results
            st.session_state["last_fig"] = fig
            load_mlflow_runs.clear()
        except Exception as exc:
            st.error(f"Fehler beim Training: {exc}")

def display_results():
    if "last_results" in st.session_state:
        r = st.session_state["last_results"]
        st.divider()
        st.subheader("Ergebnisse")
        render_metrics_row(
            labels=["MAE Modell", "MAE Naive", "R2", "Verbesserung", "Train/Test"],
            values=[f"{r['mae_primary']:.2f}", f"{r['mae_naive']:.2f}", f"{r['r2_primary']:.3f}",
                    f"{r['improvement_pct']:+.1f}%", f"{r['train_size']}/{r['test_size']}"]
        )
        render_plotly(st.session_state["last_fig"])

def render_grid_search_cv_section(pattern, store, item, season, is_daily, gap, trim):
    st.divider()
    with st.expander("Grid Search CV — AutoARIMA"):
        # Hier die vollständige Grid-Search Logik aus deinem Original einfügen
        # (Parameter-Auswahl, Button, Callback und Ergebnis-Tabelle)
        pass

def render_history():
    st.divider()
    st.subheader("MLflow Run History")
    if st.button("Aktualisieren"): load_mlflow_runs.clear()
    
    df = load_mlflow_runs()
    if df.empty:
        st.info("Noch keine Runs gefunden.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()