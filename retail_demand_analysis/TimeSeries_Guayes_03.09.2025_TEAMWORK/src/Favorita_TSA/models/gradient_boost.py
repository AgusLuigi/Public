"""
gradient_boost.py

Backend fuer GradientBoosting-Modellierung mit Lag/Rolling- und exogenen Features.
Verwendet sklearn GradientBoostingRegressor.

Funktionen
----------
  build_gb_features(df, store, item, freq, lag_periods, feature_cols,
                    trailing_zero_min_days) -> pd.DataFrame
  run_gradient_boost_plotly(df, pattern, store, item, freq, season_length,
                            test_weeks, feature_cols, lag_periods,
                            n_estimators, max_depth, learning_rate,
                            trailing_zero_min_days, img_dir) -> (dict, go.Figure)

Alle Runs werden automatisch in MLflow geloggt.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive

from Favorita_TSA.models.baseline import load_and_prepare, train_test_split
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.mlflow_utils import setup_mlflow

_CATEGORICAL_COLS = {"store_type", "family"}

_DEFAULT_LAG_PERIODS_DAILY: list[int] = [1, 7, 14, 28]
_DEFAULT_LAG_PERIODS_WEEKLY: list[int] = [1, 4, 13, 26]


def build_gb_features(
    df: pd.DataFrame,
    store: int,
    item: int,
    freq: str = "D",
    lag_periods: list[int] | None = None,
    feature_cols: list[str] | None = None,
    trailing_zero_min_days: int = 0,
) -> pd.DataFrame:
    """
    Baut das vollstaendige Feature-DataFrame fuer GradientBoosting.

    Umfasst:
    - Zeit-Features (year, month, dow/week) aus dem angereicherten DataFrame
    - Lag-Features auf unit_sales
    - Rolling-Mittelwerte und Standardabweichung auf unit_sales
    - Optionale exogene Features (inkl. One-hot Encoding)

    Parameters
    ----------
    df : DataFrame
        Angereicherter Segment-DataFrame (aus load_sarimax_segment()).
    store, item : int
    freq : str
        "D" oder "W".
    lag_periods : list[int] | None
        Lag-Perioden fuer unit_sales. Standard: [1,7,14,28] (daily) / [1,4,13,26] (weekly).
    feature_cols : list[str] | None
        Exogene Feature-Spalten.
    trailing_zero_min_days : int
        Mindest-Laenge des Null-Blocks am Ende fuer Trimming.

    Returns
    -------
    DataFrame mit Spalten [ds, y, feature_1, ...].
    Erste max(lag_periods) Zeilen werden wegen NaN-Lags gedroppt.
    """
    is_weekly = freq == "W"

    if lag_periods is None:
        lag_periods = (
            _DEFAULT_LAG_PERIODS_WEEKLY if is_weekly else _DEFAULT_LAG_PERIODS_DAILY
        )
    if feature_cols is None:
        feature_cols = []

    # Store/Item filtern
    if "store_nbr" in df.columns:
        slice_df = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
    else:
        slice_df = df.copy()

    date_col = "week_start" if is_weekly else "date"
    slice_df[date_col] = pd.to_datetime(slice_df[date_col])
    slice_df = slice_df.sort_values(date_col).reset_index(drop=True)

    # Trailing-Zero-Trimming
    if trailing_zero_min_days > 0 and not is_weekly:
        sales = slice_df["unit_sales"].values
        last_nonzero = len(sales)
        for i in range(len(sales) - 1, -1, -1):
            if sales[i] > 0:
                last_nonzero = i + 1
                break
        if len(sales) - last_nonzero >= trailing_zero_min_days:
            slice_df = slice_df.iloc[:last_nonzero].copy()
            slice_df = slice_df.reset_index(drop=True)

    # Basis: Datum + Target
    result = pd.DataFrame()
    result["ds"] = slice_df[date_col].values
    result["y"] = slice_df["unit_sales"].fillna(0).values

    # Kalender-Features direkt aus Datum berechnen (nie aus slice_df uebernehmen)
    dates = pd.to_datetime(result["ds"])
    result["year"] = dates.dt.year.astype(int)
    result["month"] = dates.dt.month.astype(int)
    if not is_weekly:
        result["dow"] = dates.dt.dayofweek.astype(int)
    result["week"] = dates.dt.isocalendar().week.astype(int)

    # Lag-Features
    for lag in sorted(lag_periods):
        result[f"sales_lag_{lag}"] = result["y"].shift(lag)

    # Rolling-Features (shift(1) verhindert Datenleck)
    result["sales_rolling_mean_7"] = (
        result["y"].shift(1).rolling(7, min_periods=1).mean()
    )
    result["sales_rolling_mean_28"] = (
        result["y"].shift(1).rolling(28, min_periods=1).mean()
    )
    result["sales_rolling_std_7"] = (
        result["y"].shift(1).rolling(7, min_periods=1).std().fillna(0)
    )

    # Exogene Features
    if feature_cols:
        cols_to_add = [c for c in feature_cols if c in slice_df.columns]
        if is_weekly:
            cols_to_add = [c for c in cols_to_add if c != "is_weekend"]

        if cols_to_add:
            x_exog = slice_df[cols_to_add].copy()

            cat_present = [c for c in _CATEGORICAL_COLS if c in x_exog.columns]
            if cat_present:
                x_exog = pd.get_dummies(x_exog, columns=cat_present, drop_first=True)

            x_exog = (
                x_exog.astype(object)
                .apply(lambda s: pd.to_numeric(s, errors="coerce"))
                .fillna(0)
                .astype(float)
            )

            for col in x_exog.columns:
                result[col] = x_exog[col].values

    # NaN-Zeilen am Anfang wegen Lags droppen
    max_lag = max(lag_periods) if lag_periods else 0
    result = result.iloc[max_lag:].reset_index(drop=True)
    result = result.dropna().reset_index(drop=True)

    # Sicherstellen dass alle Feature-Spalten numerisch sind (Period/datetime -> 0)
    for col in list(result.columns):
        if col == "ds":
            continue
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0)

    return result


def _next_gb_run_number(pattern: str) -> int:
    """Gibt die naechste Run-Nummer fuer das Pattern zurueck."""
    try:
        existing = mlflow.search_runs(
            filter_string=f"params.pattern = '{pattern}'",
            max_results=1000,
        )
        return len(existing) + 1
    except Exception:
        return 1


def run_gradient_boost_plotly(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    test_weeks: int = 4,
    feature_cols: list[str] | None = None,
    lag_periods: list[int] | None = None,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    trailing_zero_min_days: int = 0,
    img_dir: Path | str | None = None,
) -> tuple[dict, go.Figure]:
    """
    Vollstaendige GradientBoosting-Pipeline: Feature Engineering -> Split ->
    GBM fitten -> Naive Benchmark -> Metriken -> Plotly-Chart -> MLflow loggen.

    Parameters
    ----------
    df : DataFrame
        Angereicherter Store-Item-Segment-DataFrame (aus load_sarimax_segment()).
    pattern : str
    store, item : int
    freq : str
        "D" oder "W".
    season_length : int
        Saisonalitaets-Periode fuer SeasonalNaive-Benchmark.
    test_weeks : int
        Anzahl Wochen im Test-Set.
    feature_cols : list[str] | None
        Exogene Feature-Spalten.
    lag_periods : list[int] | None
        Lag-Perioden fuer unit_sales.
    n_estimators : int
        Anzahl Boosting-Stufen.
    max_depth : int
        Maximale Baumtiefe.
    learning_rate : float
        Lernrate / Schrumpfungsfaktor.
    img_dir : Path | str | None
        Verzeichnis fuer HTML-Plot-Export.

    Returns
    -------
    (results_dict, fig)
    """
    if feature_cols is None:
        feature_cols = []
    is_weekly = freq == "W"
    if lag_periods is None:
        lag_periods = (
            _DEFAULT_LAG_PERIODS_WEEKLY if is_weekly else _DEFAULT_LAG_PERIODS_DAILY
        )

    print(
        f"GradientBoosting auf {pattern} (store={store}, item={item})  "
        f"n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}"
    )

    # 1. Zeitreihe via load_and_prepare vorbereiten (inkl. korrektem Trailing-Zero-Trimming)
    ts = load_and_prepare(
        df,
        store=store,
        item=item,
        freq=freq,
        trailing_zero_min_days=trailing_zero_min_days,
    )
    ts["ds"] = pd.to_datetime(ts["ds"])
    ts_date_min = ts["ds"].min()
    ts_date_max = ts["ds"].max()

    # 2. Feature-DataFrame bauen (internes Trimming deaktiviert - ts-Bereich ist massgeblich)
    feat_df = build_gb_features(
        df,
        store=store,
        item=item,
        freq=freq,
        lag_periods=lag_periods,
        feature_cols=feature_cols,
        trailing_zero_min_days=0,
    )

    if feat_df.empty:
        msg = "Feature-DataFrame ist leer - zu wenig Daten fuer Lag-Features."
        raise ValueError(msg)

    # Auf validierten Datumsbereich von load_and_prepare einschraenken
    feat_df["ds"] = pd.to_datetime(feat_df["ds"])
    feat_df = (
        feat_df[(feat_df["ds"] >= ts_date_min) & (feat_df["ds"] <= ts_date_max)]
        .copy()
        .reset_index(drop=True)
    )

    # 3. Zeitsplit (zeitbasiert, nicht per train_test_split - GBM hat eigenes Format)
    cutoff = feat_df["ds"].max() - pd.Timedelta(weeks=test_weeks)
    train_df = feat_df[feat_df["ds"] <= cutoff].copy()
    test_df = feat_df[feat_df["ds"] > cutoff].copy()

    feature_names = [
        c
        for c in feat_df.columns
        if c not in ("ds", "y") and pd.api.types.is_numeric_dtype(feat_df[c])
    ]
    non_numeric = [
        f"{c}({feat_df[c].dtype})"
        for c in feat_df.columns
        if c not in ("ds", "y") and not pd.api.types.is_numeric_dtype(feat_df[c])
    ]
    if non_numeric:
        print(f"GBM: dropping non-numeric columns: {non_numeric}")
    x_train = train_df[feature_names].fillna(0).astype(float).values
    y_train = train_df["y"].fillna(0).astype(float).values
    x_test = test_df[feature_names].fillna(0).astype(float).values
    y_true = test_df["y"].values

    print(
        f"  Train: {len(train_df)} Zeilen, Test: {len(test_df)} Zeilen, Features: {len(feature_names)}"
    )

    # 3. GBM trainieren
    gbm = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
    )
    gbm.fit(x_train, y_train)
    pred = np.maximum(gbm.predict(x_test), 0)

    # 5. SeasonalNaive Benchmark (nutzt ts aus Schritt 1)
    train_ts, test_ts = train_test_split(ts, test_weeks=test_weeks)

    sf = StatsForecast(
        models=[SeasonalNaive(season_length=season_length)],
        freq=freq,
    )
    sf.fit(train_ts[["unique_id", "ds", "y"]].copy())
    naive_pred = sf.predict(h=len(test_ts))
    naive_vals = np.maximum(naive_pred["SeasonalNaive"].values, 0)

    # Laengen angleichen (GBM-test kann minimal kuerzer sein)
    n = min(len(y_true), len(naive_vals))
    y_true_aligned = y_true[:n]
    pred_aligned = pred[:n]
    naive_aligned = naive_vals[:n]
    test_dates = test_df["ds"].values[:n]

    # 5. Metriken
    mae_primary = float(mean_absolute_error(y_true_aligned, pred_aligned))
    mae_naive = float(mean_absolute_error(y_true_aligned, naive_aligned))
    r2_primary = (
        float(r2_score(y_true_aligned, pred_aligned))
        if len(y_true_aligned) > 1
        else float("nan")
    )
    improvement_pct = (
        (mae_naive - mae_primary) / mae_naive * 100 if mae_naive > 0 else 0.0
    )

    print(
        f"  MAE GBM={mae_primary:.3f}  MAE Naive={mae_naive:.3f}  "
        f"Verbesserung={improvement_pct:+.1f}%"
    )

    results = {
        "mae_primary": mae_primary,
        "mae_naive": mae_naive,
        "r2_primary": r2_primary,
        "improvement_pct": improvement_pct,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "feature_importances": gbm.feature_importances_.tolist(),
    }

    # 6. Plotly-Chart: Zeitreihe (2 Subplots) + Feature Importance
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=[
            "Gesamtzeitreihe",
            "Test-Zeitraum (Zoom)",
            "Feature Importance (Top 15)",
        ],
        vertical_spacing=0.09,
        row_heights=[0.33, 0.33, 0.34],
    )

    fig.add_trace(
        go.Scatter(
            x=train_df["ds"],
            y=train_df["y"],
            mode="lines",
            name="Train",
            line={"color": "#4C8BF5", "width": 1},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_true_aligned,
            mode="lines",
            name="Test (Ist)",
            line={"color": "#34A853", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=pred_aligned,
            mode="lines",
            name="GradientBoosting",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_true_aligned,
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
            x=test_dates,
            y=pred_aligned,
            mode="lines",
            name="GradientBoosting",
            line={"color": "#FF6D00", "width": 2, "dash": "dash"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=naive_aligned,
            mode="lines",
            name="Seasonal Naive",
            line={"color": "#9E9E9E", "width": 1.5, "dash": "dot"},
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Feature Importance (Top 15)
    importances = gbm.feature_importances_
    top_n = min(15, len(feature_names))
    top_idx = np.argsort(importances)[-top_n:][::-1]
    fig.add_trace(
        go.Bar(
            x=[feature_names[i] for i in top_idx],
            y=[float(importances[i]) for i in top_idx],
            name="Importance",
            marker_color="#4C8BF5",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    title = (
        f"GradientBoosting - {pattern} | Store {store} - Item {item} | "
        f"MAE={mae_primary:.2f} ({improvement_pct:+.1f}%) | "
        f"{len(feature_names)} Features"
    )
    fig.update_layout(
        title=title,
        height=950,
        legend={"orientation": "h", "y": -0.04},
    )

    # 7. MLflow loggen
    experiment = getattr(cfg.mlflow, "gb_experiment", "favorita_gb_store_item")
    setup_mlflow(experiment)

    run_n = _next_gb_run_number(pattern)
    run_name = f"{pattern}_{run_n:03d}_gradient_boost"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "pattern": pattern,
                "model_type": "gradient_boost",
                "store": store,
                "item": item,
                "freq": freq,
                "season_length": season_length,
                "test_weeks": test_weeks,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "lag_periods": str(lag_periods),
                "n_features": len(feature_names),
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
