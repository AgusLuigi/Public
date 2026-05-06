"""
sarimax.py

Backend für SARIMAX-Modellierung mit exogenen Features.
Verwendet statsmodels.tsa.statespace.sarimax.SARIMAX mit manuell
eingestellten Parametern (p,d,q)(P,D,Q,s).

Funktionen
──────────
  run_sarimax_plotly(df, pattern, store, item, freq, season_length,
                     test_weeks, order, seasonal_order, feature_cols,
                     img_dir) -> (dict, go.Figure)

Alle Runs werden automatisch in MLflow geloggt.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from itertools import product
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from statsmodels.tsa.statespace.sarimax import SARIMAX

from Favorita_TSA.models.baseline import load_and_prepare, train_test_split

# Kategorische Spalten, die one-hot encodiert werden müssen
_CATEGORICAL_COLS = {"store_type", "family"}

# Konstante Spalten, die für SARIMAX ungeeignet sind
_EXCLUDE_COLS = {
    "store_nbr",
    "item_nbr",
    "date",
    "week_start",
    "unit_sales",
    "ds",
    "y",
    "unique_id",
    "year",
    "dow",
    "year_iso",
    "week",
    "month",
}


def _build_exog_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    is_weekly: bool,
) -> pd.DataFrame:
    """
    Baut die exogene Feature-Matrix aus dem angereicherten DataFrame.

    Kategorische Spalten (store_type, family) werden one-hot encodiert.
    Bei wöchentlicher Zeitachse wird ``is_weekend`` automatisch ausgeschlossen.

    Parameters
    ----------
    df : DataFrame
        Angereicherter Store-Item-Slice (bereits gefiltert auf store/item).
    feature_cols : list[str]
        Auswahl der Feature-Spalten.
    is_weekly : bool
        Falls True, wird ``is_weekend`` aus ``feature_cols`` entfernt.

    Returns
    -------
    DataFrame mit rein numerischen Spalten, Index kompatibel mit df.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if is_weekly:
        cols = [c for c in cols if c != "is_weekend"]

    if not cols:
        return pd.DataFrame(index=df.index)

    X = df[cols].copy()

    # One-hot encoding für kategorische Spalten
    cat_present = [c for c in _CATEGORICAL_COLS if c in X.columns]
    if cat_present:
        X = pd.get_dummies(X, columns=cat_present, drop_first=True)

    # Sicherstellen: alle Spalten numerisch
    # astype(object) auf dem gesamten DataFrame entfernt nullable BooleanDtype / IntDtype,
    # danach apply(to_numeric) + fillna + astype(float) ohne column-by-column Zuweisung.
    X = (
        X.astype(object)
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .fillna(0)
        .astype(float)
    )

    return X


def _prepare_exog_aligned(
    df: pd.DataFrame,
    ts: pd.DataFrame,
    store: int,
    item: int,
    feature_cols: list[str],
    freq: str,
) -> np.ndarray | None:
    """
    Baut die exogene Feature-Matrix und aligniert sie mit der Zeitreihe *ts*.

    Returns
    -------
    np.ndarray mit Shape (len(ts), n_features) oder None wenn keine Features.
    """
    is_weekly = freq == "W"
    if not feature_cols:
        return None

    if "store_nbr" in df.columns:
        slice_df = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
        date_col = "week_start" if is_weekly else "date"
        slice_df[date_col] = pd.to_datetime(slice_df[date_col])
        slice_df = slice_df.sort_values(date_col).reset_index(drop=True)
    else:
        slice_df = df.copy()
        date_col = "week_start" if "week_start" in df.columns else "date"

    X_full = _build_exog_matrix(slice_df, feature_cols, is_weekly=is_weekly)

    if X_full.shape[1] == 0:
        return None

    ts_reset = ts.reset_index(drop=True)
    ts_ds = pd.to_datetime(ts_reset["ds"])

    if len(ts_reset) == len(X_full):
        return X_full.values

    # Merge ueber Datum
    ts_with_idx = pd.DataFrame({"ds": ts_ds})
    ts_with_idx["_pos"] = np.arange(len(ts_with_idx))
    slice_with_idx = pd.DataFrame(
        {date_col: pd.to_datetime(slice_df[date_col])}
    ).reset_index(drop=True)
    slice_with_idx["_slice_pos"] = np.arange(len(slice_with_idx))
    merged = ts_with_idx.merge(
        slice_with_idx, left_on="ds", right_on=date_col, how="left"
    )
    valid_pos = merged["_slice_pos"].fillna(-1).astype(int).values
    X_arr = np.zeros((len(ts_reset), X_full.shape[1]))
    for i, pos in enumerate(valid_pos):
        if 0 <= pos < len(X_full):
            X_arr[i] = X_full.iloc[pos].values
    return X_arr


def _next_sarimax_run_number(pattern: str) -> int:
    """Gibt die nächste Run-Nummer für das Pattern zurück."""
    try:
        existing = mlflow.search_runs(
            filter_string=f"params.pattern = '{pattern}'",
            max_results=1000,
        )
        return len(existing) + 1
    except Exception:
        return 1


def run_sarimax_plotly(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    test_weeks: int = 4,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    feature_cols: list[str] | None = None,
    trailing_zero_min_days: int = 0,
    img_dir: Path | str | None = None,
) -> tuple[dict, go.Figure]:
    """
    Vollständige SARIMAX-Pipeline: Vorbereitung → Split → SARIMAX fitten
    → Naive Benchmark → Metriken berechnen → Plotly-Chart → MLflow loggen.

    Parameters
    ----------
    df : DataFrame
        Angereicherter Store-Item-Segment-DataFrame (aus load_sarimax_segment()).
    pattern : str
        z.B. "daily_smooth", "weekly_erratic".
    store, item : int
    freq : str
        "D" für täglich, "W" für wöchentlich.
    season_length : int
        Saisonalitäts-Periode (s im seasonal_order).
    test_weeks : int
        Anzahl Wochen im Test-Set.
    order : tuple (p, d, q)
        Nicht-saisonale ARIMA-Ordnung.
    seasonal_order : tuple (P, D, Q, s)
        Saisonale ARIMA-Ordnung inkl. Periode s.
    feature_cols : list[str] | None
        Auswahl der exogenen Features. None oder leer = reines SARIMA.
    img_dir : Path | str | None
        Verzeichnis für HTML-Plot-Export.

    Returns
    -------
    (results_dict, fig)
    """
    if feature_cols is None:
        feature_cols = []

    # ── 1. Zeitreihe vorbereiten (ds/y/unique_id) ────────────────────────────
    ts = load_and_prepare(
        df,
        store=store,
        item=item,
        freq=freq,
        trailing_zero_min_days=trailing_zero_min_days,
    )

    # ── 2. Exogene Feature-Matrix aus angereichertem df ─────────────────────
    ts["ds"] = pd.to_datetime(ts["ds"])
    X_aligned = _prepare_exog_aligned(df, ts, store, item, feature_cols, freq)
    use_exog = X_aligned is not None

    # ── 3. Train/Test-Split ───────────────────────────────────────────────────
    train_ts, test_ts = train_test_split(ts, test_weeks=test_weeks)

    n_train = len(train_ts)
    if use_exog:
        X_train = X_aligned[:n_train]
        X_test = X_aligned[n_train:]
    else:
        X_train = None
        X_test = None

    # ── 4. SARIMAX fitten ─────────────────────────────────────────────────────
    print(
        f"Fitting SARIMAX{order}x{seasonal_order} "
        f"auf {pattern} (store={store}, item={item})"
    )
    print(f"  Exogene Features: {feature_cols if use_exog else 'keine'}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            endog=train_ts["y"].values,
            exog=X_train if use_exog else None,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)

    # ── 5. Forecast ───────────────────────────────────────────────────────────
    n_test = len(test_ts)
    forecast = result.forecast(steps=n_test, exog=X_test if use_exog else None)
    forecast = np.maximum(forecast, 0)  # keine negativen Verkäufe

    # ── 6. SeasonalNaive Benchmark ───────────────────────────────────────────
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=season_length)],
        freq=freq,
    )
    train_sf = train_ts[["unique_id", "ds", "y"]].copy()
    sf.fit(train_sf)
    naive_pred = sf.predict(h=n_test)
    naive_vals = np.maximum(naive_pred["SeasonalNaive"].values, 0)

    # ── 7. Metriken ───────────────────────────────────────────────────────────
    y_true = test_ts["y"].values
    mae_primary = float(mean_absolute_error(y_true, forecast))
    mae_naive = float(mean_absolute_error(y_true, naive_vals))
    r2_primary = float(r2_score(y_true, forecast)) if len(y_true) > 1 else float("nan")
    improvement_pct = (
        (mae_naive - mae_primary) / mae_naive * 100 if mae_naive > 0 else 0.0
    )

    print(
        f"  MAE SARIMAX={mae_primary:.3f}  MAE Naive={mae_naive:.3f}  "
        f"Verbesserung={improvement_pct:+.1f}%"
    )

    results = {
        "mae_primary": mae_primary,
        "mae_naive": mae_naive,
        "r2_primary": r2_primary,
        "improvement_pct": improvement_pct,
        "train_size": len(train_ts),
        "test_size": n_test,
        "n_exog": X_aligned.shape[1] if use_exog else 0,
    }

    # ── 8. Plotly-Chart ───────────────────────────────────────────────────────
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Gesamtzeitreihe", "Test-Zeitraum (Zoom)"],
        vertical_spacing=0.12,
        shared_xaxes=False,
    )

    # Gesamte Trainingsdaten
    fig.add_trace(
        go.Scatter(
            x=train_ts["ds"],
            y=train_ts["y"],
            mode="lines",
            name="Train",
            line={"color": "#4C8BF5", "width": 1},
        ),
        row=1,
        col=1,
    )
    # Test-Ist-Werte
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=test_ts["y"],
            mode="lines",
            name="Test (Ist)",
            line={"color": "#34A853", "width": 2},
        ),
        row=1,
        col=1,
    )
    # SARIMAX-Forecast
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=forecast,
            mode="lines",
            name=f"SARIMAX{order}",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )

    # Zoom: Test-Zeitraum
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=test_ts["y"],
            mode="lines",
            name="Test (Ist)",
            line={"color": "#34A853", "width": 2},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=forecast,
            mode="lines",
            name=f"SARIMAX{order}",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=naive_vals,
            mode="lines",
            name="Seasonal Naive",
            line={"color": "#9E9E9E", "width": 1.5, "dash": "dot"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    p, d, q = order
    P, D, Q, s = seasonal_order
    feat_str = ", ".join(feature_cols[:3]) + ("…" if len(feature_cols) > 3 else "")
    title = (
        f"SARIMAX({p},{d},{q})({P},{D},{Q},{s}) — {pattern} "
        f"| Store {store} · Item {item} | "
        f"MAE={mae_primary:.2f} ({improvement_pct:+.1f}%)"
    )
    if feat_str:
        title += f" | Features: {feat_str}"

    fig.update_layout(
        title=title,
        height=700,
        legend={"orientation": "h", "y": -0.08},
    )

    # ── 9. MLflow loggen ──────────────────────────────────────────────────────
    run_n = _next_sarimax_run_number(pattern)
    run_name = f"{pattern}_{run_n:03d}_sarimax"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "pattern": pattern,
                "model_type": "sarimax",
                "store": store,
                "item": item,
                "freq": freq,
                "season_length": season_length,
                "test_weeks": test_weeks,
                "p": p,
                "d": d,
                "q": q,
                "s_p": P,
                "s_d": D,
                "s_q": Q,
                "s": s,
                "n_exog_features": X_aligned.shape[1] if use_exog else 0,
                "feature_cols": str(feature_cols),
            }
        )
        mlflow.log_metrics(
            {
                "mae_primary": mae_primary,
                "mae_naive": mae_naive,
                "r2_primary": r2_primary if not np.isnan(r2_primary) else -999.0,
                "improvement_pct": improvement_pct,
            }
        )

        if img_dir is not None:
            img_path = Path(img_dir) / f"{run_name}.html"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(img_path))
            mlflow.log_artifact(str(img_path))

    return results, fig


# ─── Walk-Forward Cross-Validation ───────────────────────────────────────────


def _walk_forward_cv(
    endog: np.ndarray,
    exog: np.ndarray | None,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    horizon: int,
    n_windows: int,
    step_size: int,
) -> list[float]:
    """
    Manuelle Walk-Forward-CV fuer statsmodels SARIMAX.

    Returns
    -------
    Liste der per-Fold-MAE-Werte (np.nan bei Fehlern).
    """
    T = len(endog)
    mae_per_fold: list[float] = []
    min_train = max(2 * seasonal_order[3], 30)

    for i in range(n_windows):
        cutoff = T - horizon - (n_windows - 1 - i) * step_size
        if cutoff < min_train:
            mae_per_fold.append(np.nan)
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    endog=endog[:cutoff],
                    exog=exog[:cutoff] if exog is not None else None,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=50)

            forecast = result.forecast(
                steps=horizon,
                exog=exog[cutoff : cutoff + horizon] if exog is not None else None,
            )
            forecast = np.maximum(forecast, 0)
            y_true = endog[cutoff : cutoff + horizon]
            mae_per_fold.append(float(mean_absolute_error(y_true, forecast)))
        except Exception:
            mae_per_fold.append(np.nan)

    return mae_per_fold


# ─── Grid Search (Stufe 1: Parameter) ───────────────────────────────────────


_DEFAULT_SARIMAX_GRID: dict[str, list[int]] = {
    "p": [0, 1, 2],
    "d": [0, 1],
    "q": [0, 1, 2],
    "P": [0, 1],
    "D": [0, 1],
    "Q": [0, 1],
}


def run_sarimax_grid_search(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    horizon: int = 28,
    n_windows: int = 3,
    step_size: int | None = None,
    param_grid: dict[str, list[int]] | None = None,
    feature_cols: list[str] | None = None,
    trailing_zero_min_days: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Grid Search ueber (p,d,q)(P,D,Q) mit Walk-Forward-CV.

    Sucht die besten SARIMAX-Parameter bei fixen exogenen Features.
    Jede Kombination wird als separater MLflow-Run geloggt.

    Returns
    -------
    DataFrame mit einer Zeile pro Kombination, sortiert nach cv_mae_mean.
    Spalten: p, d, q, P, D, Q, cv_mae_mean, cv_mae_std,
             n_successful_folds, run_id, best.
    """
    param_grid = param_grid or _DEFAULT_SARIMAX_GRID
    step_size = step_size or horizon
    if feature_cols is None:
        feature_cols = []

    experiment_name = "favorita_sarimax_store_item"
    mlflow.set_experiment(experiment_name)
    _client = MlflowClient()
    _exp = mlflow.get_experiment_by_name(experiment_name)

    # Zeitreihe einmalig vorbereiten
    ts = load_and_prepare(
        df, store, item, freq=freq, trailing_zero_min_days=trailing_zero_min_days
    )
    ts["ds"] = pd.to_datetime(ts["ds"])
    endog = ts["y"].values

    # Exog einmalig bauen
    X_aligned = _prepare_exog_aligned(df, ts, store, item, feature_cols, freq)

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    total = len(combos)
    print(
        f"\nSARIMAX Grid Search: {total} Kombinationen x {n_windows} Folds "
        f"(horizon={horizon})"
    )

    gs_group = f"{pattern}_sarimax_gs"
    results: list[dict] = []

    for idx, combo_values in enumerate(combos):
        combo = dict(zip(keys, combo_values, strict=True))
        p, d, q = combo.get("p", 1), combo.get("d", 1), combo.get("q", 1)
        P, D, Q = combo.get("P", 0), combo.get("D", 0), combo.get("Q", 0)
        s_order = (P, D, Q, season_length)

        combo_label = f"p{p}d{d}q{q}P{P}D{D}Q{Q}"
        run_name = f"{gs_group}_{idx + 1:03d}_{combo_label}"
        print(f"  [{idx + 1}/{total}] {combo_label} ...", end=" ", flush=True)

        _run_id = _client.create_run(
            experiment_id=_exp.experiment_id,
            run_name=run_name,
        ).info.run_id

        try:
            mae_folds = _walk_forward_cv(
                endog=endog,
                exog=X_aligned,
                order=(p, d, q),
                seasonal_order=s_order,
                horizon=horizon,
                n_windows=n_windows,
                step_size=step_size,
            )

            valid_maes = [m for m in mae_folds if not np.isnan(m)]
            n_ok = len(valid_maes)

            if n_ok == 0 or n_ok < n_windows / 2:
                print("FAILED (zu wenige Folds konvergiert)")
                _client.set_terminated(_run_id, "FAILED")
                if progress_callback is not None:
                    progress_callback(idx + 1, total)
                continue

            cv_mae_mean = float(np.mean(valid_maes))
            cv_mae_std = float(np.std(valid_maes)) if n_ok > 1 else 0.0
            print(f"MAE={cv_mae_mean:.3f} +/- {cv_mae_std:.3f} ({n_ok}/{n_windows})")

            base_params = {
                "pattern": pattern,
                "model_type": "sarimax",
                "freq": freq,
                "store": store,
                "item": item,
                "season_length": season_length,
                "cv_horizon": horizon,
                "cv_n_windows": n_windows,
                "cv_step_size": step_size,
                "cv_group": gs_group,
                "p": p,
                "d": d,
                "q": q,
                "s_p": P,
                "s_d": D,
                "s_q": Q,
                "s": season_length,
                "n_exog_features": X_aligned.shape[1] if X_aligned is not None else 0,
                "feature_cols": str(feature_cols),
            }
            for key, val in base_params.items():
                _client.log_param(_run_id, key, str(val))
            _client.log_metric(_run_id, "cv_mae_mean", cv_mae_mean)
            _client.log_metric(_run_id, "cv_mae_std", cv_mae_std)
            for fold_idx, mae_val in enumerate(mae_folds):
                if not np.isnan(mae_val):
                    _client.log_metric(_run_id, "cv_mae_fold", mae_val, step=fold_idx)
            _client.set_tag(_run_id, "cv", "grid_search")
            _client.set_terminated(_run_id, "FINISHED")

            results.append(
                {
                    **combo,
                    "cv_mae_mean": cv_mae_mean,
                    "cv_mae_std": cv_mae_std,
                    "n_successful_folds": n_ok,
                    "run_id": _run_id,
                }
            )

        except Exception as exc:
            print(f"FAILED ({exc})")
            _client.set_terminated(_run_id, "FAILED")

        if progress_callback is not None:
            progress_callback(idx + 1, total)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("cv_mae_mean").reset_index(drop=True)
    results_df["best"] = False
    results_df.loc[0, "best"] = True

    best_run_id = results_df.loc[0, "run_id"]
    _client.set_tag(best_run_id, "best_in_group", gs_group)
    print(
        f"\nBeste Kombination: "
        f"{results_df.loc[0, list(keys)].to_dict()}  "
        f"MAE={results_df.loc[0, 'cv_mae_mean']:.3f}"
    )

    return results_df


# ─── Feature Search (Stufe 2: Feature-Gruppen) ──────────────────────────────


_DEFAULT_FEATURE_GROUPS: dict[str, list[str]] = {
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


def _get_all_candidate_features(freq: str) -> list[str]:
    """Alle verfuegbaren Einzel-Features (ohne Kategorien die one-hot werden)."""
    all_feats: list[str] = []
    for cols in _DEFAULT_FEATURE_GROUPS.values():
        for c in cols:
            if freq == "W" and c == "is_weekend":
                continue
            all_feats.append(c)
    return all_feats


def run_sarimax_feature_search(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    horizon: int = 28,
    n_windows: int = 3,
    step_size: int | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    candidate_features: list[str] | None = None,
    trailing_zero_min_days: int = 0,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """
    Stufe 2: Backward Elimination Feature Selection bei fixen SARIMAX-Parametern.

    Startet mit allen verfuegbaren Features (bekannt guter Startpunkt).
    In jedem Schritt wird das Feature entfernt, dessen Wegfall den MAE am meisten
    verbessert. Stopp wenn keine weitere Entfernung noch hilft.

    Gesamtaufwand: 1 + N*(N+1)/2 Evaluierungen (ca. 277 fuer 23 Features).

    Parameters
    ----------
    candidate_features : list[str] | None
        Einzelne Feature-Namen die getestet werden sollen.
        None = alle verfuegbaren Features.
    progress_callback : callable(done: int, total: int, msg: str) | None
        Fortschritts-Callback mit Status-Nachricht.

    Returns
    -------
    DataFrame mit Baseline + Schritten (ein Eintrag pro Schritt), sortiert nach
    step. Spalten: step, removed_feature, feature_set, cv_mae_mean, cv_mae_std,
    n_features, n_successful_folds, run_id, best.
    """
    step_size = step_size or horizon
    if candidate_features is None:
        candidate_features = _get_all_candidate_features(freq)

    experiment_name = "favorita_sarimax_store_item"
    mlflow.set_experiment(experiment_name)
    _client = MlflowClient()
    _exp = mlflow.get_experiment_by_name(experiment_name)

    ts = load_and_prepare(
        df, store, item, freq=freq, trailing_zero_min_days=trailing_zero_min_days
    )
    ts["ds"] = pd.to_datetime(ts["ds"])
    endog = ts["y"].values

    selected = list(candidate_features)  # start with ALL features
    fs_group = f"{pattern}_sarimax_fs"
    results: list[dict] = []
    step_counter = 0

    n_feats = len(selected)
    total_evals = (
        1 + n_feats * (n_feats + 1) // 2
    )  # Baseline + alle moeglichen Schritte
    eval_counter = 0

    print(
        f"\nSARIMAX Backward Elimination: {n_feats} Features, "
        f"max {total_evals} Evaluierungen (order={order}, seasonal={seasonal_order})"
    )

    def _eval_feature_set(
        feat_list: list[str], label: str
    ) -> tuple[float, float, int, str]:
        """Evaluiert ein Feature-Set, loggt in MLflow, gibt (mae, std, n_ok, run_id)."""
        X_aligned = _prepare_exog_aligned(df, ts, store, item, feat_list, freq)
        run_name = f"{fs_group}_{step_counter:02d}_{label}"
        _run_id = _client.create_run(
            experiment_id=_exp.experiment_id,
            run_name=run_name,
        ).info.run_id

        try:
            mae_folds = _walk_forward_cv(
                endog=endog,
                exog=X_aligned,
                order=order,
                seasonal_order=seasonal_order,
                horizon=horizon,
                n_windows=n_windows,
                step_size=step_size,
            )
            valid_maes = [m for m in mae_folds if not np.isnan(m)]
            n_ok = len(valid_maes)

            if n_ok == 0 or n_ok < n_windows / 2:
                _client.set_terminated(_run_id, "FAILED")
                return float("inf"), 0.0, 0, _run_id

            cv_mae_mean = float(np.mean(valid_maes))
            cv_mae_std = float(np.std(valid_maes)) if n_ok > 1 else 0.0

            params = {
                "pattern": pattern,
                "model_type": "sarimax",
                "freq": freq,
                "store": store,
                "item": item,
                "season_length": season_length,
                "cv_horizon": horizon,
                "cv_n_windows": n_windows,
                "cv_group": fs_group,
                "p": order[0],
                "d": order[1],
                "q": order[2],
                "s_p": seasonal_order[0],
                "s_d": seasonal_order[1],
                "s_q": seasonal_order[2],
                "s": seasonal_order[3],
                "n_exog_features": len(feat_list),
                "feature_cols": str(feat_list),
                "fs_step": step_counter,
            }
            for key, val in params.items():
                _client.log_param(_run_id, key, str(val))
            _client.log_metric(_run_id, "cv_mae_mean", cv_mae_mean)
            _client.log_metric(_run_id, "cv_mae_std", cv_mae_std)
            _client.set_tag(_run_id, "cv", "feature_search")
            _client.set_terminated(_run_id, "FINISHED")

            return cv_mae_mean, cv_mae_std, n_ok, _run_id

        except Exception:
            _client.set_terminated(_run_id, "FAILED")
            return float("inf"), 0.0, 0, _run_id

    # ── Baseline: MIT ALLEN Features ──────────────────────────────────────
    print(f"  [Baseline] alle {len(selected)} Features ...", end=" ", flush=True)
    base_mae, base_std, base_ok, base_rid = _eval_feature_set(
        list(selected), "baseline"
    )
    eval_counter += 1
    if progress_callback is not None:
        progress_callback(
            eval_counter, total_evals, f"Baseline ({len(selected)} Features)"
        )

    if base_ok > 0:
        print(f"MAE={base_mae:.3f}")
        results.append(
            {
                "step": 0,
                "removed_feature": "(baseline)",
                "cv_mae_mean": base_mae,
                "cv_mae_std": base_std,
                "n_features": len(selected),
                "feature_set": list(selected),
                "n_successful_folds": base_ok,
                "run_id": base_rid,
            }
        )
    best_mae = base_mae

    # ── Backward Elimination: entferne solange ein Feature den MAE verbessert
    while selected:
        step_counter += 1
        print(
            f"\n  [Step {step_counter}] Teste Entfernung von {len(selected)} Features ..."
        )
        best_to_remove = None
        best_mae_after = float("inf")
        best_std_after = 0.0
        best_ok_after = 0
        best_rid_after = ""

        for feat in selected:
            trial = [f for f in selected if f != feat]
            label = feat.replace(" ", "_")[:20]
            print(f"    - {feat} ...", end=" ", flush=True)
            mae, std, n_ok, rid = _eval_feature_set(trial, f"s{step_counter}_{label}")
            eval_counter += 1
            if progress_callback is not None:
                progress_callback(
                    eval_counter, total_evals, f"Step {step_counter}: -{feat}"
                )

            if n_ok > 0:
                print(f"MAE={mae:.3f}")
            else:
                print("FAILED")

            if mae < best_mae_after:
                best_to_remove = feat
                best_mae_after = mae
                best_std_after = std
                best_ok_after = n_ok
                best_rid_after = rid

        # Stopp wenn keine Entfernung den MAE verbessert
        if best_to_remove is None or best_mae_after >= best_mae:
            print(
                f"\n  Stopp: Keine Entfernung verbessert MAE (aktuell {best_mae:.3f})"
            )
            break

        selected.remove(best_to_remove)
        best_mae = best_mae_after
        print(
            f"  -> Entfernt: {best_to_remove} "
            f"(neuer MAE={best_mae_after:.3f}, verbleibend: {len(selected)})"
        )

        results.append(
            {
                "step": step_counter,
                "removed_feature": best_to_remove,
                "cv_mae_mean": best_mae_after,
                "cv_mae_std": best_std_after,
                "n_features": len(selected),
                "feature_set": list(selected),
                "n_successful_folds": best_ok_after,
                "run_id": best_rid_after,
            }
        )

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).reset_index(drop=True)
    results_df["best"] = False
    best_idx = results_df["cv_mae_mean"].idxmin()
    results_df.loc[best_idx, "best"] = True

    best_run_id = results_df.loc[best_idx, "run_id"]
    _client.set_tag(best_run_id, "best_in_group", fs_group)

    best_feats = results_df.loc[best_idx, "feature_set"]
    print(
        f"\nBestes Feature-Set (Step {int(results_df.loc[best_idx, 'step'])}, "
        f"{len(best_feats)} Features): {best_feats or ['keine']}  "
        f"MAE={results_df.loc[best_idx, 'cv_mae_mean']:.3f}"
    )

    return results_df
