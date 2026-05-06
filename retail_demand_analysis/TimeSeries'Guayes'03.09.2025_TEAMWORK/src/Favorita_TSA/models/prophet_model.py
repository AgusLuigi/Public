"""
prophet_model.py

Backend fuer Prophet-Modellierung mit optionalen exogenen Regressoren.
Verwendet Facebook Prophet mit manuellem Train/Test-Split.

Funktionen
----------
  run_prophet_plotly(df, pattern, store, item, freq, season_length,
                     test_weeks, feature_cols, changepoint_prior_scale,
                     seasonality_prior_scale, trailing_zero_min_days,
                     img_dir) -> (dict, go.Figure)

Alle Runs werden automatisch in MLflow geloggt.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from Favorita_TSA.models.baseline import load_and_prepare, train_test_split
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow

_CATEGORICAL_COLS = {"store_type", "family"}


def _build_regressor_df(
    df: pd.DataFrame,
    store: int,
    item: int,
    feature_cols: list[str],
    freq: str,
) -> pd.DataFrame | None:
    """
    Baut DataFrame mit Datum + Regressor-Spalten fuer Prophet.

    Returns
    -------
    DataFrame mit Spalten [ds, feat_1, feat_2, ...] oder None wenn keine Features.
    """
    is_weekly = freq == "W"
    cols = [c for c in feature_cols if c in df.columns]
    if is_weekly:
        cols = [c for c in cols if c != "is_weekend"]
    if not cols:
        return None

    if "store_nbr" in df.columns:
        slice_df = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
    else:
        slice_df = df.copy()

    date_col = "week_start" if is_weekly else "date"
    slice_df[date_col] = pd.to_datetime(slice_df[date_col])
    slice_df = slice_df.sort_values(date_col).reset_index(drop=True)

    available = [c for c in cols if c in slice_df.columns]
    if not available:
        return None

    X = slice_df[[date_col, *available]].copy()

    cat_present = [c for c in _CATEGORICAL_COLS if c in X.columns]
    if cat_present:
        X = pd.get_dummies(X, columns=cat_present, drop_first=True)

    feat_cols = [c for c in X.columns if c != date_col]
    X[feat_cols] = (
        X[feat_cols]
        .astype(object)
        .apply(lambda s: pd.to_numeric(s, errors="coerce"))
        .fillna(0)
        .astype(float)
    )

    X = X.rename(columns={date_col: "ds"})
    X["ds"] = pd.to_datetime(X["ds"])
    return X


def _next_prophet_run_number(pattern: str) -> int:
    """Gibt die naechste Run-Nummer fuer das Pattern zurueck."""
    try:
        existing = mlflow.search_runs(
            filter_string=f"params.pattern = '{pattern}'",
            max_results=1000,
        )
        return len(existing) + 1
    except Exception:
        return 1


def run_prophet_plotly(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    test_weeks: int = 4,
    feature_cols: list[str] | None = None,
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    trailing_zero_min_days: int = 0,
    img_dir: Path | str | None = None,
) -> tuple[dict, go.Figure]:
    """
    Vollstaendige Prophet-Pipeline: Vorbereitung -> Split -> Prophet fitten
    -> Naive Benchmark -> Metriken berechnen -> Plotly-Chart -> MLflow loggen.

    Parameters
    ----------
    df : DataFrame
        Angereicherter Store-Item-Segment-DataFrame (aus load_sarimax_segment()).
    pattern : str
        z.B. "daily_smooth", "weekly_erratic".
    store, item : int
    freq : str
        "D" fuer taeglich, "W" fuer woechentlich.
    season_length : int
        Saisonalitaets-Periode fuer SeasonalNaive-Benchmark.
    test_weeks : int
        Anzahl Wochen im Test-Set.
    feature_cols : list[str] | None
        Auswahl der exogenen Regressoren. None oder leer = reines Prophet.
    changepoint_prior_scale : float
        Regulierungsstaerke fuer Trend-Changepoints (hoeher = flexibler).
    seasonality_prior_scale : float
        Regulierungsstaerke fuer Saisonalitaets-Komponenten.
    img_dir : Path | str | None
        Verzeichnis fuer HTML-Plot-Export.

    Returns
    -------
    (results_dict, fig)
    """
    from prophet import Prophet  # lazy import - vermeidet Startup-Delay

    if feature_cols is None:
        feature_cols = []

    is_weekly = freq == "W"

    # 1. Zeitreihe vorbereiten
    ts = load_and_prepare(
        df,
        store=store,
        item=item,
        freq=freq,
        trailing_zero_min_days=trailing_zero_min_days,
    )
    ts["ds"] = pd.to_datetime(ts["ds"])

    # 2. Regressor-DataFrame aufbauen
    reg_df = _build_regressor_df(df, store, item, feature_cols, freq)
    regressor_cols: list[str] = []
    if reg_df is not None:
        regressor_cols = [c for c in reg_df.columns if c != "ds"]

    # 3. Train/Test-Split
    train_ts, test_ts = train_test_split(ts, test_weeks=test_weeks)

    # 4. Prophet-Training-DataFrame zusammenbauen
    train_prophet = train_ts[["ds", "y"]].copy()
    if reg_df is not None:
        train_prophet = train_prophet.merge(reg_df, on="ds", how="left")
        for col in regressor_cols:
            train_prophet[col] = train_prophet[col].fillna(0)

    # 5. Prophet-Modell bauen und fitten
    print(
        f"Fitting Prophet auf {pattern} (store={store}, item={item})  "
        f"Regressoren: {regressor_cols or 'keine'}"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            yearly_seasonality="auto",
            weekly_seasonality=not is_weekly,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        for col in regressor_cols:
            m.add_regressor(col)
        m.fit(train_prophet)

    # 6. Future DataFrame fuer Test-Periode
    future = test_ts[["ds"]].copy()
    future["ds"] = pd.to_datetime(future["ds"])
    if reg_df is not None:
        future = future.merge(reg_df, on="ds", how="left")
        for col in regressor_cols:
            future[col] = future[col].fillna(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = m.predict(future)

    pred = np.maximum(forecast["yhat"].values, 0)

    # 7. SeasonalNaive Benchmark
    sf = StatsForecast(
        models=[SeasonalNaive(season_length=season_length)],
        freq=freq,
    )
    sf.fit(train_ts[["unique_id", "ds", "y"]].copy())
    naive_pred = sf.predict(h=len(test_ts))
    naive_vals = np.maximum(naive_pred["SeasonalNaive"].values, 0)

    # 8. Metriken
    y_true = test_ts["y"].values
    mae_primary = float(mean_absolute_error(y_true, pred))
    mae_naive = float(mean_absolute_error(y_true, naive_vals))
    r2_primary = float(r2_score(y_true, pred)) if len(y_true) > 1 else float("nan")
    improvement_pct = (
        (mae_naive - mae_primary) / mae_naive * 100 if mae_naive > 0 else 0.0
    )

    print(
        f"  MAE Prophet={mae_primary:.3f}  MAE Naive={mae_naive:.3f}  "
        f"Verbesserung={improvement_pct:+.1f}%"
    )

    results = {
        "mae_primary": mae_primary,
        "mae_naive": mae_naive,
        "r2_primary": r2_primary,
        "improvement_pct": improvement_pct,
        "train_size": len(train_ts),
        "test_size": len(test_ts),
        "n_regressors": len(regressor_cols),
    }

    # 9. Plotly-Chart
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Gesamtzeitreihe", "Test-Zeitraum (Zoom)"],
        vertical_spacing=0.12,
        shared_xaxes=False,
    )

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
    fig.add_trace(
        go.Scatter(
            x=test_ts["ds"],
            y=pred,
            mode="lines",
            name="Prophet",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
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
            y=pred,
            mode="lines",
            name="Prophet",
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

    feat_str = ", ".join(feature_cols[:3]) + ("..." if len(feature_cols) > 3 else "")
    title = (
        f"Prophet - {pattern} | Store {store} - Item {item} | "
        f"MAE={mae_primary:.2f} ({improvement_pct:+.1f}%)"
    )
    if feat_str:
        title += f" | Regressoren: {feat_str}"

    fig.update_layout(
        title=title,
        height=700,
        legend={"orientation": "h", "y": -0.08},
    )

    # 10. MLflow loggen
    experiment = getattr(
        cfg.mlflow, "prophet_experiment", "favorita_prophet_store_item"
    )
    setup_mlflow(experiment)

    run_n = _next_prophet_run_number(pattern)
    run_name = f"{pattern}_{run_n:03d}_prophet"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "pattern": pattern,
                "model_type": "prophet",
                "store": store,
                "item": item,
                "freq": freq,
                "season_length": season_length,
                "test_weeks": test_weeks,
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_prior_scale": seasonality_prior_scale,
                "n_exog_features": len(regressor_cols),
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
