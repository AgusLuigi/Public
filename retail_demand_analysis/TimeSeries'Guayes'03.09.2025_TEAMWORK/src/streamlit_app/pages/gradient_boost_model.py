"""
gradient_boost_model.py

Interaktive Streamlit-Seite fuer GradientBoosting-Modellierung.
Lag-Features, exogene Feature-Gruppen und GBM-Hyperparameter
koennen interaktiv konfiguriert werden.

Voraussetzung:
  - data/metrics/*.parquet vorhanden (Forecastability-Matrizen)
  - data/processed/*.parquet vorhanden (Oil, Transactions, Stores, Items)
"""

from __future__ import annotations

import os

import mlflow
import pandas as pd
import streamlit as st

from Favorita_TSA.features.sarima_features import load_sarimax_segment
from Favorita_TSA.models.gradient_boost import run_gradient_boost_plotly
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Gradient Boosting Model")
set_plotly_theme()

os.chdir(MLRUNS_DIR.parent.parent)  # PROJECT_ROOT

EXPERIMENT = getattr(cfg.mlflow, "gb_experiment", "favorita_gb_store_item")

_pe = cfg.defaults.pattern_examples
PATTERN_DEFAULTS: dict[str, dict] = {
    "daily_smooth": vars(_pe.daily_smooth),
    "daily_erratic": vars(_pe.daily_erratic),
    "weekly_smooth": vars(_pe.weekly_smooth),
    "weekly_erratic": vars(_pe.weekly_erratic),
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature-Gruppen
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "Holidays": ["is_holiday_or_event", "pre_holiday", "post_holiday"],
    "Oil Price": [
        "oil_price",
        "oil_price_ma7",
        "oil_price_ma28",
        "oil_price_pct_change",
    ],
    "Calendar": [
        "is_weekend",
        "is_payday",
        "is_month_start",
        "is_month_end",
        "days_to_next_holiday",
        "days_since_last_holiday",
    ],
    "Transactions": ["transactions", "transactions_ma7", "transactions_z_score"],
    "Promotion": ["onpromotion", "promo_streak", "promo_rate_7d"],
    "Store / Item": ["store_cluster", "perishable", "store_type", "family"],
}

# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────


@st.cache_data(
    show_spinner="Lade und enriche Segment-Daten (einmalig, kann ~60 s dauern) ..."
)
def _load_segment(pattern: str) -> pd.DataFrame:
    return load_sarimax_segment(pattern)


@st.cache_data(ttl=15, show_spinner=False)
def _load_mlflow_runs() -> pd.DataFrame:
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
        "params.store",
        "params.item",
        "params.freq",
        "params.n_estimators",
        "params.max_depth",
        "params.learning_rate",
        "params.lag_periods",
        "params.n_features",
        "metrics.mae_primary",
        "metrics.mae_naive",
        "metrics.r2_primary",
        "metrics.improvement_pct",
        "start_time",
    ]
    existing = [c for c in wanted if c in runs.columns]
    return runs[existing].head(30)


# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Layout
# ─────────────────────────────────────────────────────────────────────────────

st.title("Gradient Boosting Model")
st.caption(
    "sklearn GradientBoostingRegressor mit Lag/Rolling-Features und optionalen "
    "exogenen Features - alle Runs werden in MLflow und img/mlflow/ gespeichert."
)

st.divider()

# ── Daten-Auswahl ─────────────────────────────────────────────────────────────

st.subheader("Daten")

col_pat, col_store, col_item, col_test, col_trim = st.columns(5)

with col_pat:
    pattern = st.selectbox(
        "Demand Pattern",
        options=list(PATTERN_DEFAULTS.keys()),
        index=0,
    )

defaults = PATTERN_DEFAULTS[pattern]
is_daily = "daily" in pattern
is_weekly = not is_daily
_default_season = defaults["season"]

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

with col_test:
    test_weeks = st.slider(
        "Test Weeks",
        min_value=1,
        max_value=52,
        value=defaults["test_weeks"],
        help="Anzahl der Wochen im Test-Set",
    )

with col_trim:
    trim_days = st.slider(
        "Trim Zero-Phase",
        min_value=0,
        max_value=90,
        value=0,
        step=5,
        format="%d Tage",
        disabled=is_weekly,
        help="Entfernt den letzten langen Null-Block. Nur fuer Daily relevant.",
    )
trailing_zero_min_days = int(trim_days) if is_daily else 0

st.divider()

# ── Lag-Features ──────────────────────────────────────────────────────────────

st.subheader("Lag-Features")
st.caption(
    "Welche vergangenen Perioden als Eingabe-Features verwendet werden. "
    "Taeglich: Lags in Tagen. Woechentlich: Lags in Wochen."
)

_default_lags_daily = [1, 7, 14, 28]
_default_lags_weekly = [1, 4, 13, 26]
_all_lags_daily = [1, 2, 3, 7, 14, 21, 28, 56]
_all_lags_weekly = [1, 2, 4, 8, 13, 26, 52]

lag_periods = st.multiselect(
    "Lag-Perioden",
    options=_all_lags_daily if is_daily else _all_lags_weekly,
    default=_default_lags_daily if is_daily else _default_lags_weekly,
    help="Auswahl der Lag-Perioden. Mindestens eine Periode erforderlich.",
)
if not lag_periods:
    lag_periods = _default_lags_daily if is_daily else _default_lags_weekly
    st.warning("Mindestens eine Lag-Periode erforderlich - Standard wird verwendet.")

st.caption(
    "Automatisch ergaenzt: `sales_rolling_mean_7`, `sales_rolling_mean_28`, "
    "`sales_rolling_std_7` (jeweils mit 1-Perioden-Shift, kein Datenleck)."
)

st.divider()

# ── Exogene Features ──────────────────────────────────────────────────────────

st.subheader("Exogene Features")
st.caption(
    "Waehle zusaetzliche exogene Features. "
    "Bei woechentlichem Pattern wird `is_weekend` automatisch ausgeblendet."
)

selected_features: list[str] = []

with st.expander("Feature-Gruppen auswaehlen", expanded=True):
    for group_name, group_cols in FEATURE_GROUPS.items():
        display_cols = (
            group_cols if is_daily else [c for c in group_cols if c != "is_weekend"]
        )
        if not display_cols:
            continue

        def _make_select_all_callback(gname: str, cols: list[str]):
            def _cb():
                new_val = st.session_state[f"gb_all_{gname}"]
                for feat in cols:
                    st.session_state[f"gb_feat_{gname}_{feat}"] = new_val

            return _cb

        g_col1, g_col2 = st.columns([1, 5])
        with g_col1:
            st.checkbox(
                "Alle",
                key=f"gb_all_{group_name}",
                value=False,
                on_change=_make_select_all_callback(group_name, display_cols),
                help=f"Alle Features der Gruppe '{group_name}' auswaehlen",
            )
        with g_col2:
            st.markdown(f"**{group_name}**")
            feat_cols_ui = st.columns(min(len(display_cols), 4))
            for i, feat in enumerate(display_cols):
                with feat_cols_ui[i % len(feat_cols_ui)]:
                    checked = st.checkbox(feat, key=f"gb_feat_{group_name}_{feat}")
                    if checked:
                        selected_features.append(feat)

st.caption(
    f"Ausgewaehlte exogene Features ({len(selected_features)}): "
    + (", ".join(selected_features) if selected_features else "- keine -")
)

st.divider()

# ── GBM-Hyperparameter ────────────────────────────────────────────────────────

st.subheader("GradientBoosting Hyperparameter")

h_col1, h_col2, h_col3, h_col4 = st.columns(4)

with h_col1:
    n_estimators = st.slider(
        "n_estimators",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Anzahl der Boosting-Stufen (Baeume).",
    )

with h_col2:
    max_depth = st.slider(
        "max_depth",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Maximale Tiefe jedes Entscheidungsbaums.",
    )

with h_col3:
    learning_rate = st.slider(
        "learning_rate",
        min_value=0.01,
        max_value=0.3,
        value=0.05,
        step=0.01,
        format="%.2f",
        help="Lernrate / Schrumpfungsfaktor pro Schritt.",
    )

with h_col4:
    s_val = st.slider(
        "s (Saison-Benchmark)",
        min_value=1,
        max_value=52,
        value=_default_season,
        help="Saisonalitaets-Periode fuer SeasonalNaive-Benchmark.",
    )

st.divider()

# ── Run-Button ────────────────────────────────────────────────────────────────

run_button = st.button(
    "Run GradientBoosting", type="primary", use_container_width=False
)

# ── Training ──────────────────────────────────────────────────────────────────

if run_button:
    setup_mlflow(EXPERIMENT)
    with st.spinner(
        f"Trainiere GradientBoosting auf {pattern} (store={store}, item={item}) ..."
    ):
        try:
            df = _load_segment(pattern)
            results, fig = run_gradient_boost_plotly(
                df=df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(s_val),
                test_weeks=int(test_weeks),
                feature_cols=selected_features,
                lag_periods=sorted(lag_periods),
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                trailing_zero_min_days=trailing_zero_min_days,
                img_dir=IMG_DIR,
            )
            st.session_state["gb_results"] = results
            st.session_state["gb_fig"] = fig
            _load_mlflow_runs.clear()
        except Exception as exc:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            st.error(f"Fehler beim GradientBoosting-Training: {exc}")
            st.exception(exc)

# ── Ergebnisse ────────────────────────────────────────────────────────────────

if "gb_results" in st.session_state:
    r = st.session_state["gb_results"]
    fig = st.session_state["gb_fig"]

    st.divider()
    st.subheader("Ergebnisse")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("MAE GBM", f"{r['mae_primary']:.2f}")
    m2.metric("MAE Naive", f"{r['mae_naive']:.2f}")
    m3.metric("R2 GBM", f"{r['r2_primary']:.3f}")
    m4.metric(
        "Verbesserung",
        f"{r['improvement_pct']:+.1f}%",
        delta=f"{r['improvement_pct']:.1f}%",
        delta_color="normal",
    )
    m5.metric("Train / Test", f"{r['train_size']} / {r['test_size']}")
    m6.metric("Features gesamt", r["n_features"])

    st.plotly_chart(fig, use_container_width=True)

# ── MLflow Run History ────────────────────────────────────────────────────────

st.divider()
st.subheader("MLflow Run History - GradientBoosting")

btn_col, _ = st.columns([1, 5])
with btn_col:
    if st.button("Aktualisieren"):
        _load_mlflow_runs.clear()

runs_df = _load_mlflow_runs()

if runs_df.empty:
    st.info("Noch keine GradientBoosting-Runs gefunden. Starte ein Modell oben.")
else:
    rename_map = {
        "tags.mlflow.runName": "Run Name",
        "params.pattern": "Pattern",
        "params.store": "Store",
        "params.item": "Item",
        "params.freq": "Freq",
        "params.n_estimators": "n_est",
        "params.max_depth": "depth",
        "params.learning_rate": "lr",
        "params.lag_periods": "Lags",
        "params.n_features": "Features #",
        "metrics.mae_primary": "MAE (GBM)",
        "metrics.mae_naive": "MAE (Naive)",
        "metrics.r2_primary": "R2",
        "metrics.improvement_pct": "Improvement %",
        "start_time": "Zeitpunkt",
    }
    display_df = runs_df.rename(columns=rename_map)

    for col in ["MAE (GBM)", "MAE (Naive)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else "-"
            )
    if "R2" in display_df.columns:
        display_df["R2"] = display_df["R2"].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "-"
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
