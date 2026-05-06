"""
sarimax_model.py

Interaktive Streamlit-Seite fuer SARIMAX-Modellierung mit exogenen Features.
Manuelle Einstellung aller ARIMA-Parameter (p,d,q)(P,D,Q,s).
Feature-Gruppen koennen per Checkbox selektiert werden.
Taegliche/woechentliche Features werden entsprechend aggregiert.

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
from Favorita_TSA.models.sarimax import (
    run_sarimax_feature_search,
    run_sarimax_grid_search,
    run_sarimax_plotly,
)
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow
from Favorita_TSA.utils.paths import IMG_DIR, MLRUNS_DIR
from Favorita_TSA.viz.ploty_theme import set_plotly_theme

# ─────────────────────────────────────────────────────────────────────────────
# Seiten-Konfiguration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="SARIMAX Model")
set_plotly_theme()

os.chdir(MLRUNS_DIR.parent.parent)  # PROJECT_ROOT

EXPERIMENT = cfg.mlflow.sarimax_experiment

_pe = cfg.defaults.pattern_examples
PATTERN_DEFAULTS: dict[str, dict] = {
    "daily_smooth": vars(_pe.daily_smooth),
    "daily_erratic": vars(_pe.daily_erratic),
    "weekly_smooth": vars(_pe.weekly_smooth),
    "weekly_erratic": vars(_pe.weekly_erratic),
}

# ─────────────────────────────────────────────────────────────────────────────
# Feature-Gruppen (Name -> Liste der Spalten)
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
        "is_weekend",  # nur taeglich
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


@st.cache_data(ttl=60, show_spinner=False)
def _load_gs_from_mlflow(
    pattern: str, store_nbr: int, item_nbr: int, cv_tag: str
) -> pd.DataFrame | None:
    """Laedt vorhandene Grid-Search- oder Feature-Search-Ergebnisse aus MLflow."""
    setup_mlflow(EXPERIMENT)
    try:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT],
            filter_string=(
                f"tags.cv = '{cv_tag}' "
                f"and params.pattern = '{pattern}' "
                f"and params.store = '{store_nbr}' "
                f"and params.item = '{item_nbr}'"
            ),
            order_by=["metrics.cv_mae_mean ASC"],
        )
    except Exception:
        return None

    if runs.empty:
        return None

    if cv_tag == "grid_search":
        param_cols = {
            "params.p": "p",
            "params.d": "d",
            "params.q": "q",
            "params.s_p": "P",
            "params.s_d": "D",
            "params.s_q": "Q",
        }
        result = pd.DataFrame()
        for mlflow_col, col_name in param_cols.items():
            if mlflow_col in runs.columns:
                result[col_name] = pd.to_numeric(runs[mlflow_col], errors="coerce")
        if "metrics.cv_mae_mean" in runs.columns:
            result["cv_mae_mean"] = runs["metrics.cv_mae_mean"]
        if "metrics.cv_mae_std" in runs.columns:
            result["cv_mae_std"] = runs["metrics.cv_mae_std"]
        result["run_id"] = runs["run_id"]
        result = result.dropna(subset=["cv_mae_mean"]).reset_index(drop=True)
        if result.empty:
            return None
        result["best"] = False
        result.loc[0, "best"] = True
        return result

    if cv_tag == "feature_search":
        result = pd.DataFrame()
        if "params.fs_step" in runs.columns:
            result["step"] = pd.to_numeric(runs["params.fs_step"], errors="coerce")
        if "params.added_feature" in runs.columns:
            result["added_feature"] = runs["params.added_feature"]
        if "params.feature_cols" in runs.columns:
            result["feature_set"] = runs["params.feature_cols"]
        if "params.n_exog_features" in runs.columns:
            result["n_features"] = pd.to_numeric(
                runs["params.n_exog_features"], errors="coerce"
            )
        if "metrics.cv_mae_mean" in runs.columns:
            result["cv_mae_mean"] = runs["metrics.cv_mae_mean"]
        if "metrics.cv_mae_std" in runs.columns:
            result["cv_mae_std"] = runs["metrics.cv_mae_std"]
        result["run_id"] = runs["run_id"]
        result = result.dropna(subset=["cv_mae_mean"]).reset_index(drop=True)
        if result.empty:
            return None
        # Nach Step sortieren damit die Reihenfolge stimmt
        if "step" in result.columns:
            result = result.sort_values("step").reset_index(drop=True)
        result["best"] = False
        best_idx = result["cv_mae_mean"].idxmin()
        result.loc[best_idx, "best"] = True
        return result

    return None


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
        "params.p",
        "params.d",
        "params.q",
        "params.s_p",
        "params.s_d",
        "params.s_q",
        "params.s",
        "params.n_exog_features",
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

st.title("SARIMAX Model")
st.caption(
    "Manuelle SARIMA-Parameter + auswaehlbare exogene Features - "
    "alle Runs werden in MLflow und img/mlflow/ gespeichert."
)

st.divider()

# ── Zeile 1: Daten-Auswahl ───────────────────────────────────────────────────

st.subheader("Daten")

col_pat, col_store, col_item, col_test, col_trim, col_season = st.columns(6)

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
        help="Entfernt den letzten langen Null-Block am Ende der Zeitreihe. "
        "0 = deaktiviert. Nur fuer Daily-Muster relevant.",
    )
trailing_zero_min_days = int(trim_days) if is_daily else 0

with col_season:
    s_val = st.slider(
        "s (Season)",
        min_value=1,
        max_value=52,
        value=_default_season,
        help="Saisonalitaets-Periode: 7 fuer taeglich, 52 fuer woechentlich",
    )

st.divider()

# ── Vorhandene Grid-Search-Ergebnisse aus MLflow laden ────────────────────────

_gs_key = f"sarimax_gs_{pattern}_{store}_{item}"
_fs_key = f"sarimax_fs_{pattern}_{store}_{item}"

if _gs_key not in st.session_state:
    _cached_gs = _load_gs_from_mlflow(pattern, int(store), int(item), "grid_search")
    if _cached_gs is not None:
        st.session_state[_gs_key] = _cached_gs

if _fs_key not in st.session_state:
    _cached_fs = _load_gs_from_mlflow(pattern, int(store), int(item), "feature_search")
    if _cached_fs is not None:
        st.session_state[_fs_key] = _cached_fs

# ── Grid Search CV ───────────────────────────────────────────────────────────

with st.expander("Grid Search CV - SARIMAX", expanded=False):
    st.caption(
        "Stufe 1 testet alle (p,d,q)(P,D,Q)-Kombinationen mit Walk-Forward-CV "
        "unter Einbeziehung aller verfuegbaren exogenen Features (SARIMAX-Kontext). "
        "Stufe 2 fuehrt Backward Elimination durch: startet mit allen Features und "
        "entfernt schrittweise jene, deren Wegfall den MAE verbessert. "
        "Beide Stufen sind unabhaengig voneinander startbar."
    )

    _sx = cfg.models.sarimax
    gs_col1, gs_col2 = st.columns(2)
    with gs_col1:
        gs_p = st.multiselect(
            "p (AR)",
            options=[0, 1, 2, 3, 4, 5],
            default=_sx.grid_p,
            help="Nicht-saisonale AR-Ordnungen",
        )
        gs_d = st.multiselect(
            "d (Differenzierung)",
            options=[0, 1, 2],
            default=_sx.grid_d,
            help="Differenzierungsordnungen",
        )
        gs_q = st.multiselect(
            "q (MA)",
            options=[0, 1, 2, 3, 4, 5],
            default=_sx.grid_q,
            help="Nicht-saisonale MA-Ordnungen",
        )
    with gs_col2:
        gs_P = st.multiselect(
            "P (saisonal AR)",
            options=[0, 1, 2, 3],
            default=_sx.grid_P,
            help="Saisonale AR-Ordnungen",
        )
        gs_D = st.multiselect(
            "D (saisonal Diff.)",
            options=[0, 1, 2],
            default=_sx.grid_D,
            help="Saisonale Differenzierungsordnungen",
        )
        gs_Q = st.multiselect(
            "Q (saisonal MA)",
            options=[0, 1, 2, 3],
            default=_sx.grid_Q,
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
        _default_horizon = int(test_weeks) * (7 if is_daily else 1)
        gs_horizon = st.number_input(
            "Horizon (Perioden)",
            min_value=1,
            value=_default_horizon,
            help="Forecast-Horizont je Fold in Perioden (Tage oder Wochen)",
        )
    with gs_cv_col3:
        gs_n_combos = (
            len(gs_p or [1])
            * len(gs_d or [1])
            * len(gs_q or [1])
            * len(gs_P or [0])
            * len(gs_D or [0])
            * len(gs_Q or [0])
        )
        st.metric("Kombinationen (Stufe 1)", gs_n_combos)

    if gs_n_combos * gs_n_windows > 500:
        st.warning(
            f"{gs_n_combos} Kombinationen x {gs_n_windows} Folds = "
            f"{gs_n_combos * gs_n_windows} Fits - das kann lange dauern."
        )

    # Alle verfuegbaren Features fuer Stufe 1 (SARIMAX-Kontext, nicht reines SARIMA)
    _all_gs_features = [
        feat
        for group_cols in FEATURE_GROUPS.values()
        for feat in group_cols
        if not (is_weekly and feat == "is_weekend")
    ]

    # ── Zwei separate Buttons ──────────────────────────────────────────────
    _gs_btn_col, _fs_btn_col, _ = st.columns([1, 1, 2])

    with _gs_btn_col:
        gs_run = st.button(
            "Stufe 1: Parameter-Search", type="secondary", use_container_width=True
        )
    with _fs_btn_col:
        _has_gs_for_fs = (
            _gs_key in st.session_state and not st.session_state[_gs_key].empty
        )
        fs_run = st.button(
            "Stufe 2: Feature-Search",
            type="secondary",
            use_container_width=True,
            disabled=not _has_gs_for_fs,
            help=(
                "Forward Stepwise Selection ueber einzelne Features "
                "mit den besten Parametern aus Stufe 1"
                if _has_gs_for_fs
                else "Zuerst Stufe 1 ausfuehren"
            ),
        )

    # ── Stufe 1 ausfuehren ────────────────────────────────────────────────
    if gs_run:
        _gs_param_grid = {
            "p": gs_p or [1],
            "d": gs_d or [1],
            "q": gs_q or [1],
            "P": gs_P or [0],
            "D": gs_D or [0],
            "Q": gs_Q or [0],
        }
        setup_mlflow(EXPERIMENT)

        _gs_progress = st.progress(0.0, text="Stufe 1: Vorbereitung ...")
        _gs_status = st.empty()

        def _gs_callback(done: int, total: int) -> None:
            _gs_progress.progress(
                done / total, text=f"Stufe 1: Kombination {done}/{total}"
            )
            _gs_status.caption(f"Stufe 1: {done}/{total} abgeschlossen")

        try:
            _gs_df = _load_segment(pattern)
            gs_results = run_sarimax_grid_search(
                df=_gs_df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(s_val),
                horizon=int(gs_horizon),
                n_windows=int(gs_n_windows),
                param_grid=_gs_param_grid,
                feature_cols=_all_gs_features,
                trailing_zero_min_days=trailing_zero_min_days,
                progress_callback=_gs_callback,
            )
            st.session_state[_gs_key] = gs_results
            _load_mlflow_runs.clear()
            _load_gs_from_mlflow.clear()
            _gs_progress.progress(1.0, text="Stufe 1: Fertig!")
        except Exception as _exc:
            st.error(f"Stufe 1 Fehler: {_exc}")
            st.exception(_exc)

    # ── Stufe 2 ausfuehren ────────────────────────────────────────────────
    if fs_run and _has_gs_for_fs:
        _best_gs = st.session_state[_gs_key]
        _best_row = _best_gs[_best_gs["best"]].iloc[0]
        _best_order = (int(_best_row["p"]), int(_best_row["d"]), int(_best_row["q"]))
        _best_seasonal = (
            int(_best_row["P"]),
            int(_best_row["D"]),
            int(_best_row["Q"]),
            int(s_val),
        )
        setup_mlflow(EXPERIMENT)

        _fs_progress = st.progress(0.0, text="Stufe 2: Feature-Suche ...")
        _fs_status = st.empty()

        def _fs_callback(done: int, total: int, msg: str) -> None:
            _fs_progress.progress(
                min(done / max(total, 1), 1.0),
                text=f"Stufe 2: {msg} ({done}/{total})",
            )
            _fs_status.caption(f"Stufe 2: {done}/{total} - {msg}")

        try:
            _fs_df = _load_segment(pattern)
            fs_results = run_sarimax_feature_search(
                df=_fs_df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(s_val),
                horizon=int(gs_horizon),
                n_windows=int(gs_n_windows),
                order=_best_order,
                seasonal_order=_best_seasonal,
                trailing_zero_min_days=trailing_zero_min_days,
                progress_callback=_fs_callback,
            )
            st.session_state[_fs_key] = fs_results
            _load_mlflow_runs.clear()
            _load_gs_from_mlflow.clear()
            _fs_progress.progress(1.0, text="Stufe 2: Fertig!")
        except Exception as _exc:
            st.error(f"Stufe 2 Fehler: {_exc}")
            st.exception(_exc)

# ── Grid Search Ergebnisse ───────────────────────────────────────────────────

if _gs_key in st.session_state:
    _gs = st.session_state[_gs_key]
    if not _gs.empty:
        _best = _gs[_gs["best"]].iloc[0]
        _param_cols = [c for c in ["p", "d", "q", "P", "D", "Q"] if c in _best.index]
        st.success(
            "**Stufe 1 - Beste Parameter:** "
            + "  |  ".join(f"**{k}** = {int(_best[k])}" for k in _param_cols)
            + f"  ->  MAE = {_best['cv_mae_mean']:.3f}"
            + (f" +/- {_best['cv_mae_std']:.3f}" if "cv_mae_std" in _best.index else "")
        )
        _gs_display = [c for c in _gs.columns if c not in ("run_id", "feature_cols")]
        st.dataframe(
            _gs[_gs_display].style.highlight_min(
                subset=["cv_mae_mean"], color="#1a472a"
            ),
            use_container_width=True,
            hide_index=True,
        )

if _fs_key in st.session_state:
    _fs = st.session_state[_fs_key]
    if not _fs.empty:
        _best_fs = _fs[_fs["best"]].iloc[0]
        _best_feats = _best_fs.get("feature_set", [])
        if isinstance(_best_feats, list):
            _feat_display = ", ".join(_best_feats) if _best_feats else "keine"
        else:
            _feat_display = str(_best_feats) if _best_feats else "keine"
        _best_step = int(_best_fs.get("step", 0))
        _best_n = int(_best_fs.get("n_features", 0))
        st.success(
            f"**Stufe 2 - Beste Kombination (Step {_best_step}, {_best_n} Features):** "
            f"{_feat_display}"
            f"  ->  MAE = {_best_fs['cv_mae_mean']:.3f}"
            + (
                f" +/- {_best_fs['cv_mae_std']:.3f}"
                if "cv_mae_std" in _best_fs.index
                else ""
            )
        )
        _fs_display = [c for c in _fs.columns if c not in ("run_id", "feature_set")]
        st.dataframe(
            _fs[_fs_display].style.highlight_min(
                subset=["cv_mae_mean"], color="#1a472a"
            ),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ── Feature-Auswahl ──────────────────────────────────────────────────────────

st.subheader("Exogene Features")
st.caption(
    "Waehle die exogenen Variablen fuer das SARIMAX-Modell. "
    "Bei woechentlichem Pattern wird `is_weekend` automatisch ausgeblendet."
)

selected_features: list[str] = []

with st.expander("Feature-Gruppen auswaehlen", expanded=True):
    for group_name, group_cols in FEATURE_GROUPS.items():
        # Bei weekly: is_weekend ausblenden
        display_cols = (
            group_cols if is_daily else [c for c in group_cols if c != "is_weekend"]
        )
        if not display_cols:
            continue

        # on_change Callback: setzt Session State aller Feature-Checkboxen der Gruppe
        def _make_select_all_callback(gname: str, cols: list[str]):
            def _cb():
                new_val = st.session_state[f"all_{gname}"]
                for feat in cols:
                    st.session_state[f"feat_{gname}_{feat}"] = new_val

            return _cb

        g_col1, g_col2 = st.columns([1, 5])
        with g_col1:
            st.checkbox(
                "Alle",
                key=f"all_{group_name}",
                value=False,
                on_change=_make_select_all_callback(group_name, display_cols),
                help=f"Alle Features der Gruppe '{group_name}' auswaehlen",
            )
        with g_col2:
            st.markdown(f"**{group_name}**")
            feat_cols_ui = st.columns(min(len(display_cols), 4))
            for i, feat in enumerate(display_cols):
                with feat_cols_ui[i % len(feat_cols_ui)]:
                    checked = st.checkbox(feat, key=f"feat_{group_name}_{feat}")
                    if checked:
                        selected_features.append(feat)

st.caption(
    f"Ausgewaehlte Features ({len(selected_features)}): "
    + (
        ", ".join(selected_features)
        if selected_features
        else "- keine (reines SARIMA) -"
    )
)

st.divider()

# ── Zeile 3: SARIMAX Parameter (manuell) ─────────────────────────────────────

st.subheader("SARIMAX Parameter (p,d,q)(P,D,Q,s)")

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    p_val = st.number_input(
        "p", min_value=0, max_value=5, value=1, help="AR-Ordnung (nicht-saisonal)"
    )
with c2:
    d_val = st.number_input(
        "d", min_value=0, max_value=2, value=1, help="Differenzierungsordnung"
    )
with c3:
    q_val = st.number_input(
        "q", min_value=0, max_value=5, value=1, help="MA-Ordnung (nicht-saisonal)"
    )
with c4:
    P_val = st.number_input(
        "P", min_value=0, max_value=3, value=1, help="Saisonale AR-Ordnung"
    )
with c5:
    D_val = st.number_input(
        "D", min_value=0, max_value=2, value=0, help="Saisonale Differenzierungsordnung"
    )
with c6:
    Q_val = st.number_input(
        "Q", min_value=0, max_value=3, value=1, help="Saisonale MA-Ordnung"
    )

order = (int(p_val), int(d_val), int(q_val))
seasonal_order = (int(P_val), int(D_val), int(Q_val), int(s_val))

st.divider()

# ── Run-Buttons ──────────────────────────────────────────────────────────────

_has_gs = _gs_key in st.session_state and not st.session_state[_gs_key].empty

run_col, best_col, _ = st.columns([1, 1, 3])
with run_col:
    run_button = st.button("Run SARIMAX", use_container_width=True, type="primary")
with best_col:
    run_best_button = st.button(
        "Run with Best Parameters",
        use_container_width=True,
        type="secondary",
        disabled=not _has_gs,
        help=(
            "Nutzt die besten Parameter aus dem letzten Grid Search"
            if _has_gs
            else "Zuerst Grid Search ausfuehren"
        ),
    )

# ── Training ─────────────────────────────────────────────────────────────────


def _run_sarimax_training(
    use_order: tuple[int, int, int],
    use_seasonal: tuple[int, int, int, int],
    use_features: list[str],
) -> None:
    """Shared training logic for both run buttons."""
    setup_mlflow(EXPERIMENT)
    with st.spinner(
        f"Trainiere SARIMAX{use_order}x{use_seasonal} auf {pattern} "
        f"(store={store}, item={item}) ..."
    ):
        try:
            df = _load_segment(pattern)
            results, fig = run_sarimax_plotly(
                df=df,
                pattern=pattern,
                store=int(store),
                item=int(item),
                freq="D" if is_daily else "W",
                season_length=int(s_val),
                test_weeks=int(test_weeks),
                order=use_order,
                seasonal_order=use_seasonal,
                feature_cols=use_features,
                trailing_zero_min_days=trailing_zero_min_days,
                img_dir=IMG_DIR,
            )
            st.session_state["sarimax_results"] = results
            st.session_state["sarimax_fig"] = fig
            _load_mlflow_runs.clear()
        except Exception as exc:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            st.error(f"Fehler beim SARIMAX-Training: {exc}")
            st.exception(exc)


if run_button:
    _run_sarimax_training(order, seasonal_order, selected_features)

if run_best_button and _has_gs:
    _best_gs = st.session_state[_gs_key]
    _best_row = _best_gs[_best_gs["best"]].iloc[0]
    _best_order = (int(_best_row["p"]), int(_best_row["d"]), int(_best_row["q"]))
    _best_seasonal = (
        int(_best_row["P"]),
        int(_best_row["D"]),
        int(_best_row["Q"]),
        int(s_val),
    )
    # Beste Features aus Stufe 2 verwenden, falls vorhanden
    _best_features = selected_features
    _feat_source = "manuell"
    if _fs_key in st.session_state and not st.session_state[_fs_key].empty:
        _fs_best_row = st.session_state[_fs_key]
        _fs_best_row = _fs_best_row[_fs_best_row["best"]].iloc[0]
        _raw_feats = _fs_best_row.get("feature_set", None)
        if isinstance(_raw_feats, list):
            _best_features = _raw_feats
            _feat_source = "Stufe 2"
        elif isinstance(_raw_feats, str) and _raw_feats.startswith("["):
            import ast

            try:
                _best_features = ast.literal_eval(_raw_feats)
                _feat_source = "Stufe 2"
            except Exception:
                pass
    st.info(
        f"Beste Parameter: SARIMAX{_best_order}x{_best_seasonal}  |  "
        f"Features ({_feat_source}): {len(_best_features)}"
    )
    _run_sarimax_training(_best_order, _best_seasonal, _best_features)

# ── Ergebnisse ───────────────────────────────────────────────────────────────

if "sarimax_results" in st.session_state:
    r = st.session_state["sarimax_results"]
    fig = st.session_state["sarimax_fig"]

    st.divider()
    st.subheader("Ergebnisse")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("MAE SARIMAX", f"{r['mae_primary']:.2f}")
    m2.metric("MAE Naive", f"{r['mae_naive']:.2f}")
    m3.metric("R2 SARIMAX", f"{r['r2_primary']:.3f}")
    m4.metric(
        "Verbesserung",
        f"{r['improvement_pct']:+.1f}%",
        delta=f"{r['improvement_pct']:.1f}%",
        delta_color="normal",
    )
    m5.metric("Train / Test", f"{r['train_size']} / {r['test_size']}")
    m6.metric("Exogene Features", r["n_exog"])

    st.plotly_chart(fig, use_container_width=True)

# ── MLflow Run History ────────────────────────────────────────────────────────

st.divider()
st.subheader("MLflow Run History - SARIMAX")

btn_col, _ = st.columns([1, 5])
with btn_col:
    if st.button("Aktualisieren"):
        _load_mlflow_runs.clear()

runs_df = _load_mlflow_runs()

if runs_df.empty:
    st.info("Noch keine SARIMAX-Runs gefunden. Starte ein Modell oben.")
else:
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
        "start_time": "Zeitpunkt",
    }
    display_df = runs_df.rename(columns=rename_map)

    for col in ["MAE (SARIMAX)", "MAE (Naive)"]:
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
