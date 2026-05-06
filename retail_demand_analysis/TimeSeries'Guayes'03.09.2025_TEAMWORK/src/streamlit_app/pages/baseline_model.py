"""
baseline_model.py

Interaktive Streamlit-Seite fuer Baseline-Modelltraining.
Konfiguration direkt auf der Seite (keine Sidebar).
Unterstuetzt AutoARIMA (SARIMA) und Theta mit vollem Parameter-Set.
Alle Runs werden automatisch in MLflow getracked.
Plots werden zusaetzlich in img/mlflow/ gespeichert.

Voraussetzung: Die Forecastability-Matrizen wurden bereits berechnet
               (data/metrics/*.parquet) und die Fact-Table ist vorhanden.
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd
import streamlit as st

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

# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Baseline Model")
set_plotly_theme()

os.chdir(MLRUNS_DIR.parent.parent)  # PROJECT_ROOT

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
# Daten-Caching
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Lade Daten (einmalig, kann ~60 s dauern) ...")
def _load_all_dataframes() -> dict[str, pd.DataFrame]:
    return build_dataframes()


def get_dataframe(pattern: str) -> pd.DataFrame:
    return _load_all_dataframes()[PATTERN_KEY_MAP[pattern]]


@st.cache_data(ttl=15, show_spinner=False)
def load_mlflow_runs() -> pd.DataFrame:
    setup_mlflow(EXPERIMENT)
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT],
            order_by=["start_time DESC"],
        )
    except Exception:
        return pd.DataFrame()

    wanted = [
        "run_id",
        "tags.mlflow.runName",
        "params.pattern",
        "params.model_type",
        "params.store",
        "params.item",
        "params.freq",
        "params.season_length",
        "params.test_weeks",
        "params.mp_max_p",
        "params.mp_s_max_p",
        "params.mp_max_q",
        "params.mp_s_max_q",
        "params.mp_d",
        "params.mp_seasonal",
        "params.mp_decomposition_type",
        "metrics.mae_primary",
        "metrics.mae_naive",
        "metrics.improvement_pct",
        "start_time",
    ]
    existing = [c for c in wanted if c in runs.columns]
    return runs[existing].head(30)


# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("Baseline Model")
st.caption(
    "Konfiguriere und trainiere Baseline-Modelle (AutoARIMA / Theta) — alle Runs werden in MLflow und img/mlflow/ gespeichert."
)

st.divider()

# ── Zeile 1: Daten-Auswahl ───────────────────────────────────────────────────

st.subheader("Daten")

col_pat, col_store, col_item, col_gap, col_trim = st.columns(5)

with col_pat:
    pattern = st.selectbox(
        "Demand Pattern",
        options=list(PATTERN_DEFAULTS.keys()),
        index=0,
    )

defaults = PATTERN_DEFAULTS[pattern]
is_daily = "daily" in pattern
is_weekly = not is_daily

with col_store:
    store = st.number_input(
        "Store Nr.",
        min_value=1,
        max_value=54,
        value=defaults["store"],
        step=1,
    )

with col_item:
    item = st.number_input(
        "Item Nr.",
        min_value=1,
        value=defaults["item"],
        step=1,
    )

with col_gap:
    gap_pct = st.slider(
        "Gap Threshold",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        format="%d%%",
        help="Anteil fehlender Tage (Nulltage), ab dem Dropna statt 0-Auffüllung verwendet wird (nur Daily)",
    )
    gap_threshold = gap_pct / 100

with col_trim:
    trim_days = st.slider(
        "Trim Zero-Phase",
        min_value=0,
        max_value=90,
        value=cfg.models.trailing_zero_min_days,
        step=5,
        format="%d Tage",
        disabled=is_weekly,
        help="Entfernt den letzten langen Null-Block am Ende der Zeitreihe. "
        "0 = deaktiviert. Nur für Daily-Muster relevant.",
    )
trailing_zero_min_days = int(trim_days) if is_daily else 0

st.divider()

# ── Zeile 2: Modell-Basis ────────────────────────────────────────────────────

st.subheader("Modell")

col_model, col_season, col_test = st.columns(3)

with col_model:
    model_type = st.radio(
        "Modelltyp",
        options=["sarima", "theta"],
        index=0 if defaults["model"] == "sarima" else 1,
        format_func=str.upper,
        horizontal=True,
    )

with col_season:
    season_length = st.slider(
        "Season Length",
        min_value=1,
        max_value=52,
        value=defaults["season"],
        help="Saisonalitaets-Periode: 7 fuer taeglich, 52 fuer woechentlich",
    )

with col_test:
    test_weeks = st.slider(
        "Test Weeks",
        min_value=1,
        max_value=52,
        value=defaults["test_weeks"],
        help="Anzahl der Wochen im Test-Set",
    )

st.divider()

# ── Zeile 3: Modell-spezifische Parameter ───────────────────────────────────

model_params: dict = {}

if model_type == "sarima":
    st.subheader("AutoARIMA Parameter")

    # Max-Grenzen
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    _a = cfg.models.autoarima
    with c1:
        max_p = st.number_input(
            "max_p",
            min_value=0,
            max_value=10,
            value=_a.max_p,
            help="Max. AR-Ordnung (nicht-saisonal)",
        )
    with c2:
        max_q = st.number_input(
            "max_q",
            min_value=0,
            max_value=10,
            value=_a.max_q,
            help="Max. MA-Ordnung (nicht-saisonal)",
        )
    with c3:
        max_d = st.number_input(
            "max_d",
            min_value=0,
            max_value=3,
            value=_a.max_d,
            help="Max. Differenzierungsordnung",
        )
    with c4:
        max_P = st.number_input(
            "max_P",
            min_value=0,
            max_value=5,
            value=_a.max_P,
            help="Max. saisonale AR-Ordnung",
        )
    with c5:
        max_Q = st.number_input(
            "max_Q",
            min_value=0,
            max_value=5,
            value=_a.max_Q,
            help="Max. saisonale MA-Ordnung",
        )
    with c6:
        max_D = st.number_input(
            "max_D",
            min_value=0,
            max_value=2,
            value=_a.max_D,
            help="Max. saisonale Differenzierungsordnung",
        )

    # Start-/Minimalwerte fuer die Gittersuche
    s1, s2, s3, s4, _, _ = st.columns(6)
    with s1:
        start_p = st.number_input(
            "start_p",
            min_value=0,
            max_value=10,
            value=_a.start_p,
            help="Startwert der AR-Gittersuche (nicht-saisonal)",
        )
    with s2:
        start_q = st.number_input(
            "start_q",
            min_value=0,
            max_value=10,
            value=_a.start_q,
            help="Startwert der MA-Gittersuche (nicht-saisonal)",
        )
    with s3:
        start_P = st.number_input(
            "start_P",
            min_value=0,
            max_value=5,
            value=_a.start_P,
            help="Startwert der saisonalen AR-Gittersuche",
        )
    with s4:
        start_Q = st.number_input(
            "start_Q",
            min_value=0,
            max_value=5,
            value=_a.start_Q,
            help="Startwert der saisonalen MA-Gittersuche",
        )

    c7, c8, c9, _ = st.columns(4)
    with c7:
        d_auto = st.checkbox(
            "d = auto", value=True, help="Differenzierungsordnung automatisch bestimmen"
        )
        d_val = (
            None
            if d_auto
            else st.number_input(
                "d (fix)", min_value=0, max_value=3, value=0, key="d_fix"
            )
        )
    with c8:
        D_auto = st.checkbox(
            "D = auto",
            value=True,
            help="Saisonale Differenzierungsordnung automatisch bestimmen",
        )
        D_val = (
            None
            if D_auto
            else st.number_input(
                "D (fix)", min_value=0, max_value=2, value=0, key="D_fix"
            )
        )
    with c9:
        seasonal = st.checkbox(
            "Seasonal", value=True, help="Saisonale Komponente beruecksichtigen"
        )

    model_params = {
        "max_p": int(max_p),
        "max_q": int(max_q),
        "max_d": int(max_d),
        "max_P": int(max_P),
        "max_Q": int(max_Q),
        "max_D": int(max_D),
        "start_p": int(start_p),
        "start_q": int(start_q),
        "start_P": int(start_P),
        "start_Q": int(start_Q),
        "seasonal": seasonal,
    }
    if not d_auto and d_val is not None:
        model_params["d"] = int(d_val)
    if not D_auto and D_val is not None:
        model_params["D"] = int(D_val)

else:  # theta
    st.subheader("Theta Parameter")
    c1, _ = st.columns([1, 3])
    with c1:
        decomposition_type = st.radio(
            "Decomposition Type",
            options=["multiplicative", "additive"],
            index=0,
            horizontal=True,
            help="Multiplikativ fuer Zeitreihen mit skalierenden Schwankungen,\n"
            "Additiv fuer Zeitreihen mit konstantem Saisonmuster",
        )
    model_params = {"decomposition_type": decomposition_type}

st.divider()

# ── Run-Button ───────────────────────────────────────────────────────────────

run_col, _ = st.columns([1, 4])
with run_col:
    run_button = st.button("Run Model", use_container_width=True, type="primary")

# ── Training ─────────────────────────────────────────────────────────────────

if run_button:
    setup_mlflow(EXPERIMENT)

    with st.spinner(
        f"Trainiere {model_type.upper()} auf {pattern} (store={store}, item={item}) ..."
    ):
        try:
            df = get_dataframe(pattern)

            if is_weekly:
                df = aggregate_to_weekly(df, store=int(store), item=int(item))
                if df is None:
                    st.error(
                        f"Keine Daten fuer Store {store}, Item {item} im Pattern '{pattern}'."
                    )
                    st.stop()

            results, fig = run_baseline_plotly(
                df=df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(season_length),
                test_weeks=int(test_weeks),
                model_type=model_type,
                gap_threshold=float(gap_threshold),
                trailing_zero_min_days=trailing_zero_min_days,
                model_params=model_params,
                img_dir=IMG_DIR,
            )

            st.session_state["last_results"] = results
            st.session_state["last_fig"] = fig
            load_mlflow_runs.clear()

        except Exception as exc:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            st.error(f"Fehler beim Training: {exc}")
            st.exception(exc)

# ── Ergebnisse ───────────────────────────────────────────────────────────────

if "last_results" in st.session_state:
    r = st.session_state["last_results"]
    fig = st.session_state["last_fig"]

    st.divider()
    st.subheader("Ergebnisse")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("MAE Modell", f"{r['mae_primary']:.2f}")
    m2.metric("MAE Naive", f"{r['mae_naive']:.2f}")
    m3.metric("R2 Modell", f"{r['r2_primary']:.3f}")
    m4.metric(
        "Verbesserung",
        f"{r['improvement_pct']:+.1f}%",
        delta=f"{r['improvement_pct']:.1f}%",
        delta_color="normal",
    )
    m5.metric("Train / Test", f"{r['train_size']} / {r['test_size']}")

    st.plotly_chart(fig, use_container_width=True)

# ── Grid Search CV ───────────────────────────────────────────────────────────

st.divider()
with st.expander("Grid Search CV — AutoARIMA", expanded=False):
    st.caption(
        "Testet alle Kombinationen der gewählten Parameter mit Zeitreihen-Cross-Validation "
        "(Walk-Forward). Nur für SARIMA verfügbar. Jede Kombination wird in MLflow geloggt."
    )

    gs_col1, gs_col2 = st.columns(2)
    _a = cfg.models.autoarima
    with gs_col1:
        gs_max_p = st.multiselect(
            "max_p",
            options=[0, 1, 2, 3, 4],
            default=_a.grid_p,
            help="Nicht-saisonale AR-Ordnungen",
        )
        gs_max_q = st.multiselect(
            "max_q",
            options=[0, 1, 2, 3, 4],
            default=_a.grid_q,
            help="Nicht-saisonale MA-Ordnungen",
        )
        gs_seasonal = st.checkbox(
            "Seasonal", value=True, help="Saisonale Komponente aktivieren"
        )
    with gs_col2:
        gs_max_P = st.multiselect(
            "max_P (saisonal)",
            options=[0, 1, 2, 3],
            default=_a.grid_P,
            help="Saisonale AR-Ordnungen",
        )
        gs_max_Q = st.multiselect(
            "max_Q (saisonal)",
            options=[0, 1, 2, 3],
            default=_a.grid_Q,
            help="Saisonale MA-Ordnungen",
        )

    gs_cv_col1, gs_cv_col2, gs_cv_col3 = st.columns(3)
    with gs_cv_col1:
        gs_n_windows = st.slider(
            "CV Folds",
            min_value=2,
            max_value=5,
            value=cfg.models.cv_folds,
            help="Anzahl der Walk-Forward Folds",
        )
    with gs_cv_col2:
        _default_horizon = defaults["test_weeks"] * (7 if is_daily else 1)
        gs_horizon = st.number_input(
            "Horizon (Perioden)",
            min_value=1,
            value=_default_horizon,
            help="Forecast-Horizont je Fold in Perioden (Tage oder Wochen)",
        )
    with gs_cv_col3:
        gs_n_combos = len(
            list(
                __import__("itertools").product(
                    gs_max_p or [1],
                    gs_max_q or [1],
                    gs_max_P or [0],
                    gs_max_Q or [0],
                    [gs_seasonal],
                )
            )
        )
        st.metric("Kombinationen", gs_n_combos)

    gs_run = st.button(
        "Grid Search starten", type="secondary", disabled=(model_type != "sarima")
    )
    if model_type != "sarima":
        st.caption("Grid Search ist nur für SARIMA verfügbar.")

    if gs_run:
        _gs_param_grid = {
            "max_p": gs_max_p or [1],
            "max_q": gs_max_q or [1],
            "max_P": gs_max_P or [0],
            "max_Q": gs_max_Q or [0],
            "seasonal": [gs_seasonal],
        }
        setup_mlflow(EXPERIMENT)

        _gs_progress = st.progress(0.0, text="Vorbereitung …")
        _gs_status = st.empty()

        def _gs_callback(done: int, total: int) -> None:
            _gs_progress.progress(done / total, text=f"Kombination {done}/{total}")
            _gs_status.caption(f"{done}/{total} abgeschlossen")

        try:
            _gs_df = get_dataframe(pattern)
            if is_weekly:
                _gs_df = aggregate_to_weekly(_gs_df, store=int(store), item=int(item))
                if _gs_df is None:
                    st.error(f"Keine Daten für Store {store}, Item {item}.")
                    st.stop()

            gs_results = run_grid_search_cv(
                df=_gs_df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(season_length),
                horizon=int(gs_horizon),
                n_windows=int(gs_n_windows),
                param_grid=_gs_param_grid,
                gap_threshold=float(gap_threshold),
                trailing_zero_min_days=trailing_zero_min_days,
                progress_callback=_gs_callback,
            )
            st.session_state["gs_results"] = gs_results
            load_mlflow_runs.clear()
            _gs_progress.progress(1.0, text="Fertig!")

        except Exception as _exc:
            st.error(f"Grid Search Fehler: {_exc}")
            st.exception(_exc)

if "gs_results" in st.session_state:
    _gs = st.session_state["gs_results"]
    if not _gs.empty:
        _best = _gs[_gs["best"]].iloc[0]
        st.success(
            "Beste Parameter — "
            + "  |  ".join(
                f"**{k}** = {_best[k]}"
                for k in _gs.columns
                if k not in ("cv_mae_mean", "cv_mae_std", "run_id", "best")
            )
            + f"  →  MAE = {_best['cv_mae_mean']:.3f} ± {_best['cv_mae_std']:.3f}"
        )

        display_cols = [c for c in _gs.columns if c != "run_id"]
        st.dataframe(
            _gs[display_cols].style.highlight_min(
                subset=["cv_mae_mean"], color="#1a472a"
            ),
            use_container_width=True,
            hide_index=True,
        )

# ── MLflow Run History ────────────────────────────────────────────────────────

st.divider()
st.subheader("MLflow Run History")

btn_col, _ = st.columns([1, 5])
with btn_col:
    if st.button("Aktualisieren"):
        load_mlflow_runs.clear()

runs_df = load_mlflow_runs()

if runs_df.empty:
    st.info("Noch keine Runs gefunden. Starte ein Modell oben.")
else:
    rename_map = {
        "tags.mlflow.runName": "Run Name",
        "params.pattern": "Pattern",
        "params.model_type": "Modell",
        "params.store": "Store",
        "params.item": "Item",
        "params.freq": "Freq",
        "params.season_length": "Season",
        "params.test_weeks": "Test Weeks",
        "params.mp_max_p": "max_p",
        "params.mp_s_max_p": "max_P (seas.)",
        "params.mp_max_q": "max_q",
        "params.mp_s_max_q": "max_Q (seas.)",
        "params.mp_d": "d",
        "params.mp_seasonal": "Seasonal",
        "params.mp_decomposition_type": "Decomposition",
        "metrics.mae_primary": "MAE (Modell)",
        "metrics.mae_naive": "MAE (Naive)",
        "metrics.improvement_pct": "Improvement %",
        "start_time": "Zeitpunkt",
    }
    display_df = runs_df.rename(columns=rename_map)

    for col in ["MAE (Modell)", "MAE (Naive)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )
    if "Improvement %" in display_df.columns:
        display_df["Improvement %"] = display_df["Improvement %"].map(
            lambda x: f"{x:+.1f}%" if pd.notna(x) else "-"
        )
    if "Zeitpunkt" in display_df.columns:
        display_df["Zeitpunkt"] = pd.to_datetime(display_df["Zeitpunkt"]).dt.strftime(
            "%Y-%m-%d %H:%M"
        )

    cols_to_show = [c for c in display_df.columns if c != "run_id"]
    st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True)
