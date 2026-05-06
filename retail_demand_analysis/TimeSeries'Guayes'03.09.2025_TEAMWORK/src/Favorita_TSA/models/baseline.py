"""
baseline.py

Shared helpers for Favorita store-item baseline modeling.
Supports both the Theta and SARIMA baseline notebooks as well as the
interactive Streamlit model-training page.

Functions
─────────
  load_and_prepare(df, store, item, freq, gap_threshold) -> DataFrame (ds/y/unique_id)
  train_test_split(df, test_weeks)                       -> (train_df, test_df)
  aggregate_to_weekly(df, store, item)                   -> DataFrame (week_start/unit_sales)
  run_baseline_plotly(df, pattern, store, item, ...)     -> (dict, go.Figure)

Quick start
-----------
  from Favorita_TSA.models.baseline import (
      load_and_prepare,
      train_test_split,
      aggregate_to_weekly,
      run_baseline_plotly,
  )

  results, fig = run_baseline_plotly(df, pattern="daily_smooth", ...)
  fig.show()  # Jupyter / notebook
  # or: st.plotly_chart(fig)  # Streamlit
"""

from __future__ import annotations

import tempfile
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
from statsforecast.models import AutoARIMA, SeasonalNaive, Theta

# ─────────────────────────────────────────────────────────────────────────────
# Allowed model-specific parameter keys (used to filter model_params)
# ─────────────────────────────────────────────────────────────────────────────

_AUTOARIMA_KEYS = {
    "d",
    "D",
    "max_p",
    "max_q",
    "max_d",
    "max_P",
    "max_Q",
    "max_D",
    "start_p",
    "start_q",
    "start_P",
    "start_Q",
    "seasonal",
    "information_criterion",
}

_THETA_KEYS = {"decomposition_type"}

# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────


def _detect_gaps(
    df: pd.DataFrame,
    date_col: str = "ds",
    freq: str = "D",
) -> dict:
    """
    Detect missing dates in a time series DataFrame.

    Returns
    -------
    dict with keys:
        has_gaps     : bool
        n_missing    : int
        pct_missing  : float  (fraction of full range that is missing)
        missing_dates: list   (first 10 missing dates)
    """
    if len(df) == 0:
        return {
            "has_gaps": False,
            "n_missing": 0,
            "pct_missing": 0.0,
            "missing_dates": [],
        }

    dates = pd.to_datetime(df[date_col])
    full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
    missing = full_range.difference(dates)

    return {
        "has_gaps": len(missing) > 0,
        "n_missing": len(missing),
        "pct_missing": len(missing) / len(full_range) if len(full_range) > 0 else 0.0,
        "missing_dates": list(missing[:10]),
    }


def _trim_trailing_zero_phase(ts: pd.DataFrame, min_days: int = 30) -> pd.DataFrame:
    """Schneidet die Zeitreihe am Beginn des letzten langen Null-Blocks ab.

    Findet ALLE zusammenhängenden Null-Blöcke >= min_days und schneidet am
    Start des LETZTEN solchen Blocks ab. Alles danach (inkl. isolierter
    Einzelverkäufe nach der langen Nullphase) wird entfernt.

    Beispiel: [...Verkäufe...][305 Nulltage][2 Verkäufe] → Ende vor den 305 Nulltagen.
    """
    if min_days <= 0 or ts.empty:
        return ts

    y = ts["y"].values
    n = len(y)

    # Alle Zero-Blöcke durchlaufen, letzten langen merken
    last_long_block_start = None
    i = 0
    while i < n:
        if y[i] == 0:
            j = i
            while j < n and y[j] == 0:
                j += 1
            if j - i >= min_days:
                last_long_block_start = i
            i = j
        else:
            i += 1

    if last_long_block_start is None:
        return ts  # kein langer Null-Block → kein Trimming

    cutoff_date = (
        ts["ds"].iloc[last_long_block_start - 1].date()
        if last_long_block_start > 0
        else ts["ds"].iloc[0].date()
    )
    removed = n - last_long_block_start
    print(
        f"   Trimming at last zero-block (>= {min_days} days): "
        f"{removed} rows removed, new end={cutoff_date}"
    )
    return ts.iloc[:last_long_block_start].reset_index(drop=True)


def _safe_param_key(k: str) -> str:
    """
    Map a statsforecast model-param key to a macOS-safe MLflow param key.

    statsforecast uses mixed-case keys for seasonal parameters:
        max_p / max_P,  max_q / max_Q,  max_d / max_D,
        start_p / start_P,  start_q / start_Q,  d / D

    On macOS (case-insensitive HFS+/APFS), "mp_max_p" and "mp_max_P" map to
    the same file in the MLflow FileStore, causing a "Changing param values"
    error when both are logged.

    Convention: keys with any uppercase letter get a "s_" prefix and are
    fully lowercased, making them unambiguous:
        max_p  → max_p      (unchanged)
        max_P  → s_max_p    (seasonal variant, prefix + lowercase)
    """
    if any(c.isupper() for c in k):
        return f"s_{k.lower()}"
    return k


def _next_run_number(pattern: str) -> int:
    """
    Query the active MLflow experiment for existing runs matching *pattern*
    and return the next sequential number.  Falls back to 1 on any error.
    """
    try:
        existing = mlflow.search_runs(
            filter_string=f"params.pattern = '{pattern}'",
            max_results=1000,
        )
        return len(existing) + 1
    except Exception:
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_and_prepare(
    df: pd.DataFrame,
    store: int,
    item: int,
    freq: str = "D",
    gap_threshold: float = 0.05,  # noqa: ARG001  # kept for API compatibility
    trailing_zero_min_days: int = 0,
) -> pd.DataFrame:
    """
    Filter a store-item from the fact table and return a statsforecast-ready
    DataFrame with columns: ds (datetime64), y (float), unique_id (str).

    Accepts either:
    - A full fact-table segment (smooth_daily, erratic_daily, ...) with a
      ``store_nbr`` column -> filters by store/item before processing.
    - A pre-filtered, weekly-aggregated DataFrame from ``aggregate_to_weekly()``
      that has no ``store_nbr`` column -> uses as-is.

    Gap handling (daily only):
      - Alle fehlenden Tage werden mit 0 aufgefüllt (fehlender Tag = kein Umsatz)

    Parameters
    ----------
    df : DataFrame
    store : int
    item : int
    freq : str
        "D" for daily, "W" for weekly.
    gap_threshold : float
        Fraction of missing dates above which interpolation is skipped.
        Default 0.05 (5 %).

    Returns
    -------
    DataFrame with columns: ds, y, unique_id
    """
    if "store_nbr" in df.columns:
        if freq == "W":
            date_col = "week_start" if "week_start" in df.columns else None
            if date_col is None:
                df = df.reset_index()
                date_col = "week_start" if "week_start" in df.columns else df.columns[0]
        else:
            date_col = "date" if "date" in df.columns else None
            if date_col is None:
                df = df.reset_index()
                date_col = "date" if "date" in df.columns else df.columns[0]

        print(f"   Using column: '{date_col}' (freq={freq})")

        ts = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
        if len(ts) == 0:
            raise ValueError(f"No data found for store={store}, item={item}")

        target_col = "unit_sales" if "unit_sales" in ts.columns else "target_sales"
        ts = ts[[date_col, target_col]].rename(
            columns={date_col: "ds", target_col: "y"}
        )
    else:
        print("   Using pre-filtered data")
        date_col = "week_start" if "week_start" in df.columns else "ds"
        target_col = "unit_sales" if "unit_sales" in df.columns else "y"
        ts = df[[date_col, target_col]].rename(
            columns={date_col: "ds", target_col: "y"}
        )

    ts["ds"] = pd.to_datetime(ts["ds"])
    ts = ts.sort_values("ds").reset_index(drop=True)
    ts["unique_id"] = f"store_{store}_item_{item}"

    print(f"Loaded: {len(ts)} obs | {ts['ds'].min().date()} to {ts['ds'].max().date()}")
    print(f"   Mean: {ts['y'].mean():.2f}, Std: {ts['y'].std():.2f}")

    n_nan = ts["y"].isna().sum()
    if n_nan > 0:
        print(f"   Removing {n_nan} NaN values")
        ts = ts.dropna(subset=["y"])

    if freq == "D":
        gap_info = _detect_gaps(ts, date_col="ds", freq="D")
        if gap_info["has_gaps"]:
            pct = gap_info["pct_missing"]
            full_range = pd.date_range(ts["ds"].min(), ts["ds"].max(), freq="D")
            ts_cont = ts.set_index("ds").reindex(full_range).reset_index()
            ts_cont = ts_cont.rename(columns={"index": "ds"})
            ts_cont["y"] = ts_cont["y"].fillna(0)
            ts_cont["unique_id"] = ts_cont["unique_id"].ffill().bfill()
            print(
                f"   Filled {gap_info['n_missing']} daily gaps ({pct:.1%}) with 0 (zero-sales days)"
            )
            ts = ts_cont
        if trailing_zero_min_days > 0:
            ts = _trim_trailing_zero_phase(ts, min_days=trailing_zero_min_days)
    else:
        print("   Weekly data ready (no gap filling needed)")

    remaining_nan = ts["y"].isna().sum()
    if remaining_nan > 0:
        print(f"   Safety fill: {remaining_nan} remaining NaN -> 0")
        ts["y"] = ts["y"].fillna(0)

    print(f"   Final: {len(ts)} observations")
    return ts[["ds", "y", "unique_id"]]


def train_test_split(
    df: pd.DataFrame,
    test_weeks: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-based train/test split.

    Cutoff = max(ds) - test_weeks * 7 days.
    Everything on or before the cutoff is train; everything after is test.

    Returns
    -------
    (train, test) -- both DataFrames with the same columns as *df*.
    """
    cutoff = df["ds"].max() - pd.Timedelta(weeks=test_weeks)
    train = df[df["ds"] <= cutoff].copy()
    test = df[df["ds"] > cutoff].copy()

    print(f"Split: {len(train)} train | {len(test)} test")
    print(f"   Train: {train['ds'].min().date()} to {train['ds'].max().date()}")
    print(f"   Test:  {test['ds'].min().date()} to {test['ds'].max().date()}")

    return train, test


def aggregate_to_weekly(
    df: pd.DataFrame,
    store: int,
    item: int,
) -> pd.DataFrame | None:
    """
    Filter one store-item from a daily fact table and aggregate to ISO weekly.

    Parameters
    ----------
    df : DataFrame
        Daily fact table with a ``store_nbr`` and ``item_nbr`` column.
    store : int
    item : int

    Returns
    -------
    DataFrame with columns: week_start, unit_sales, store_nbr, item_nbr
    Returns None if the store-item has no rows.
    """
    ts = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()

    if len(ts) == 0:
        print(f"   No data for store={store}, item={item}")
        return None

    if "date" in ts.columns:
        date_col = "date"
    elif "week_start" in ts.columns:
        date_col = "week_start"
    else:
        ts = ts.reset_index()
        date_col = "date" if "date" in ts.columns else ts.columns[0]

    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.sort_values(date_col)

    time_diffs = ts[date_col].diff().dt.days.dropna()
    median_diff = time_diffs.median()
    print(f"   Input frequency: {median_diff:.0f} day(s) between observations")

    ts["week_start"] = ts[date_col].dt.to_period("W").dt.start_time

    target_col = "unit_sales" if "unit_sales" in ts.columns else "target_sales"

    ts_weekly = (
        ts.groupby("week_start")
        .agg({target_col: "sum", "store_nbr": "first", "item_nbr": "first"})
        .reset_index()
    )

    print(
        f"   Aggregated: {len(ts)} daily obs -> {len(ts_weekly)} weekly obs "
        f"({ts_weekly['week_start'].min().date()} to {ts_weekly['week_start'].max().date()})"
    )
    return ts_weekly


def run_baseline_plotly(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    test_weeks: int = 4,
    model_type: str = "sarima",
    gap_threshold: float = 0.05,
    trailing_zero_min_days: int = 0,
    model_params: dict | None = None,
    img_dir: Path | str | None = None,
    mlflow_experiment: str | None = None,
) -> tuple[dict, go.Figure]:
    """
    Full baseline pipeline: prepare -> split -> fit primary + SeasonalNaive
    -> evaluate -> build plot -> log to MLflow -> optionally save plot to disk.

    The caller is responsible for configuring the MLflow tracking URI and
    (optionally) the experiment before calling this function.  If
    *mlflow_experiment* is provided, ``mlflow.set_experiment()`` is called
    inside this function as a convenience.

    Run names follow the pattern ``{pattern}_{NNN}_{model_type}``
    where NNN is a zero-padded sequential number across all runs for that
    pattern in the current experiment.

    Parameters
    ----------
    df : DataFrame
        Fact-table segment or pre-aggregated weekly DataFrame.
    pattern : str
        Human-readable label for this run, e.g. "daily_smooth".
    store, item : int
    freq : str
        Pandas / statsforecast frequency string ("D" or "W").
    season_length : int
        Seasonality period passed to Theta / AutoARIMA / SeasonalNaive.
    test_weeks : int
        Number of weeks to hold out as the test set.
    model_type : str
        "theta" or "sarima".
    gap_threshold : float
        Passed through to ``load_and_prepare``.
    model_params : dict | None
        Extra keyword arguments forwarded to the primary model constructor.

        For SARIMA (AutoARIMA), accepted keys:
            d, D, max_p, max_q, max_d, max_P, max_Q, max_D,
            start_p, start_q, start_P, start_Q,
            seasonal, information_criterion

        For Theta, accepted keys:
            decomposition_type  ("multiplicative" | "additive")

        Unknown keys are silently ignored.
    img_dir : Path | str | None
        Directory where the forecast plot is saved as an HTML file.
        The directory is created if it does not exist.
        File name: ``{run_name}.html``.
        If None, no file is written.
    mlflow_experiment : str | None
        Optional experiment name override.

    Returns
    -------
    (results_dict, fig)
        results_dict contains: pattern, model_type, freq, store, item,
            season_length, test_weeks, train_size, test_size,
            mae_primary, r2_primary, mae_naive, r2_naive, improvement_pct,
            plus all model_params keys prefixed with "param_".
        fig is the Plotly figure (caller decides whether to call fig.show()
            or st.plotly_chart(fig)).
    """
    print(f"\n{'=' * 70}\n{pattern.upper()} | {model_type.upper()}\n{'=' * 70}")

    experiment_name = mlflow_experiment or "favorita_baseline_store_item"
    mlflow.set_experiment(experiment_name)

    _client = MlflowClient()
    _exp = mlflow.get_experiment_by_name(experiment_name)

    run_number = _next_run_number(pattern)
    run_name = f"{pattern}_{run_number:03d}_{model_type}"
    print(f"Run name: {run_name}")

    model_params = model_params or {}

    _run_id = _client.create_run(
        experiment_id=_exp.experiment_id,
        run_name=run_name,
    ).info.run_id
    print(f"MLflow run_id: {_run_id}")

    try:
        # 1. Prepare
        ts = load_and_prepare(
            df,
            store,
            item,
            freq=freq,
            gap_threshold=gap_threshold,
            trailing_zero_min_days=trailing_zero_min_days,
        )

        # 2. Split
        train, test = train_test_split(ts, test_weeks=test_weeks)

        # 3. Primary model
        if model_type == "theta":
            theta_kwargs = {k: v for k, v in model_params.items() if k in _THETA_KEYS}
            print(
                f"\nTraining Theta (season_length={season_length}, {theta_kwargs}) ..."
            )
            model_primary = StatsForecast(
                models=[Theta(season_length=season_length, **theta_kwargs)],
                freq=freq,
                n_jobs=1,
            )
            primary_col = "Theta"
        else:
            arima_kwargs = {
                k: v for k, v in model_params.items() if k in _AUTOARIMA_KEYS
            }
            print(
                f"\nTraining AutoARIMA (season_length={season_length}, {arima_kwargs}) ..."
            )
            model_primary = StatsForecast(
                models=[AutoARIMA(season_length=season_length, **arima_kwargs)],
                freq=freq,
                n_jobs=1,
            )
            primary_col = "AutoARIMA"

        model_primary.fit(train)

        # 4. Forecast primary
        horizon = len(test)
        print(f"   Forecasting {horizon} periods ...")
        forecasts_primary = model_primary.predict(h=horizon).reset_index()

        test_primary = test.copy()
        test_primary[primary_col] = forecasts_primary[primary_col].values

        actuals = test_primary["y"].values
        preds_primary = test_primary[primary_col].values
        mask = ~np.isnan(actuals) & ~np.isnan(preds_primary)

        if mask.sum() == 0:
            raise ValueError("All primary-model predictions are NaN.")

        mae_primary = mean_absolute_error(actuals[mask], preds_primary[mask])
        r2_primary = r2_score(actuals[mask], preds_primary[mask])
        print(f"   Primary -- MAE: {mae_primary:.2f} | R2: {r2_primary:.3f}")

        # 5. Naive benchmark
        print("\nTraining SeasonalNaive ...")
        model_naive = StatsForecast(
            models=[SeasonalNaive(season_length=season_length)],
            freq=freq,
            n_jobs=1,
        )
        model_naive.fit(train)
        forecasts_naive = model_naive.predict(h=horizon).reset_index()

        test_naive = test.copy()
        test_naive["SeasonalNaive"] = forecasts_naive["SeasonalNaive"].values
        preds_naive = test_naive["SeasonalNaive"].values

        mae_naive = mean_absolute_error(actuals[mask], preds_naive[mask])
        r2_naive = r2_score(actuals[mask], preds_naive[mask])
        print(f"   Naive  -- MAE: {mae_naive:.2f} | R2: {r2_naive:.3f}")

        improvement = (
            float("nan")
            if mae_naive == 0
            else (mae_naive - mae_primary) / mae_naive * 100
        )
        print(f"\nImprovement vs Naive: {improvement:+.1f}%")

        # 6. Plot
        model_label = model_type.upper()
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"{pattern.upper()} -- {model_label} vs Naive  [{run_name}]",
                f"Test Period  (Improvement: {improvement:+.1f}%)",
            ),
            vertical_spacing=0.12,
        )
        fig.add_trace(
            go.Scatter(
                x=train["ds"], y=train["y"], mode="lines", name="Train", opacity=0.6
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_primary["ds"],
                y=test_primary["y"],
                mode="lines+markers",
                name="Actual",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_primary["ds"],
                y=test_primary[primary_col],
                mode="lines+markers",
                name=f"{model_label} (MAE={mae_primary:.1f})",
                line={"dash": "dash"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_naive["ds"],
                y=test_naive["SeasonalNaive"],
                mode="lines+markers",
                name=f"Naive (MAE={mae_naive:.1f})",
                line={"dash": "dot"},
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_primary["ds"],
                y=test_primary["y"],
                mode="lines+markers",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_primary["ds"],
                y=test_primary[primary_col],
                mode="lines+markers",
                showlegend=False,
                line={"dash": "dash"},
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_naive["ds"],
                y=test_naive["SeasonalNaive"],
                mode="lines+markers",
                showlegend=False,
                line={"dash": "dot"},
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Sales")
        fig.update_layout(height=800, hovermode="x unified")

        # 7. Save plot to disk
        if img_dir is not None:
            img_path = Path(img_dir)
            img_path.mkdir(parents=True, exist_ok=True)
            html_file = img_path / f"{run_name}.html"
            fig.write_html(str(html_file))
            print(f"   Plot saved: {html_file}")

        # 8. Build result dicts
        pred_table = test_primary[["ds", "y"]].rename(columns={"y": "actual"}).copy()
        pred_table["pred_primary"] = test_primary[primary_col].values
        pred_table["pred_naive"] = test_naive["SeasonalNaive"].values
        pred_table["error_primary"] = pred_table["actual"] - pred_table["pred_primary"]
        pred_table["error_naive"] = pred_table["actual"] - pred_table["pred_naive"]
        pred_table["pattern"] = pattern
        pred_table["model_type"] = model_type
        pred_table["store_nbr"] = store
        pred_table["item_nbr"] = item
        pred_table["freq"] = freq
        pred_table["season_length"] = season_length
        pred_table["test_weeks"] = test_weeks
        pred_table["ds"] = pd.to_datetime(pred_table["ds"]).dt.strftime("%Y-%m-%d")

        params = {
            "pattern": pattern,
            "model_type": model_type,
            "freq": freq,
            "store": store,
            "item": item,
            "season_length": season_length,
            "test_weeks": test_weeks,
            "trailing_zero_min_days": trailing_zero_min_days,
            **{f"mp_{_safe_param_key(k)}": v for k, v in model_params.items()},
        }
        metrics = {
            "train_size": len(train),
            "test_size": len(test),
            "mae_naive": float(mae_naive),
            "r2_naive": float(r2_naive),
            "mae_primary": float(mae_primary),
            "r2_primary": float(r2_primary),
            "improvement_pct": float(improvement),
        }

        # 9. Log to MLflow via client (no fluent API — avoids run-stack issues)
        for key, value in params.items():
            _client.log_param(_run_id, key, str(value))
        for key, value in metrics.items():
            _client.log_metric(_run_id, key, float(value))

        with tempfile.TemporaryDirectory() as _tmp:
            _fig_path = Path(_tmp) / "forecast.html"
            fig.write_html(str(_fig_path))
            _client.log_artifact(_run_id, str(_fig_path), artifact_path="plots")

            _tbl_path = Path(_tmp) / "predictions.json"
            pred_table.to_json(str(_tbl_path), orient="records", date_format="iso")
            _client.log_artifact(_run_id, str(_tbl_path), artifact_path="tables")

        _client.set_terminated(_run_id, "FINISHED")
        print(f"   MLflow run finished: {run_name}")

    except Exception:
        _client.set_terminated(_run_id, "FAILED")
        raise

    return {**params, **metrics}, fig


# ─────────────────────────────────────────────────────────────────────────────
# Grid Search CV
# ─────────────────────────────────────────────────────────────────────────────


def run_grid_search_cv(
    df: pd.DataFrame,
    pattern: str,
    store: int,
    item: int,
    freq: str = "D",
    season_length: int = 7,
    horizon: int = 28,
    n_windows: int = 3,
    step_size: int | None = None,
    param_grid: dict | None = None,
    gap_threshold: float = 0.05,
    trailing_zero_min_days: int = 0,
    mlflow_experiment: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """
    Grid search over AutoARIMA parameters with time-series cross-validation.

    For every combination in *param_grid*, a StatsForecast cross_validation run
    is performed and the mean MAE across all CV folds is computed.  Each
    combination is logged as a separate MLflow run (tagged ``cv=grid_search``).

    Parameters
    ----------
    df : DataFrame
        Fact-table segment or pre-aggregated weekly DataFrame.
    pattern : str
        Demand pattern label, e.g. "daily_smooth".
    store, item : int
    freq : str
        "D" or "W".
    season_length : int
        Passed to AutoARIMA.
    horizon : int
        Forecast horizon (number of periods) for each CV fold.
    n_windows : int
        Number of rolling CV windows.
    step_size : int | None
        Gap between consecutive cutoff dates.  Defaults to *horizon*.
    param_grid : dict | None
        Mapping of AutoARIMA param name → list of values to try.
        Defaults to a small sensible grid:
            {"max_p": [1, 2, 3], "max_q": [1, 2, 3],
             "max_P": [0, 1],    "max_Q": [0, 1],
             "seasonal": [True]}
    gap_threshold : float
        Passed to ``load_and_prepare``.
    mlflow_experiment : str | None
        Optional experiment name override.
    progress_callback : callable(current: int, total: int) | None
        Called after each combination is evaluated.  Useful for Streamlit
        progress bars.

    Returns
    -------
    pd.DataFrame
        One row per parameter combination, sorted by ``cv_mae_mean`` ascending.
        Columns: all param keys, ``cv_mae_mean``, ``cv_mae_std``, ``run_id``.
        The best row also has ``best=True``; all others ``best=False``.
    """
    _DEFAULT_GRID: dict[str, list] = {
        "max_p": [1, 2, 3],
        "max_q": [1, 2, 3],
        "max_P": [0, 1],
        "max_Q": [0, 1],
        "seasonal": [True],
    }
    param_grid = param_grid or _DEFAULT_GRID
    step_size = step_size or horizon

    experiment_name = mlflow_experiment or "favorita_baseline_store_item"
    mlflow.set_experiment(experiment_name)
    _client = MlflowClient()
    _exp = mlflow.get_experiment_by_name(experiment_name)

    # Prepare time series once — reused for every combo
    ts = load_and_prepare(
        df,
        store,
        item,
        freq=freq,
        gap_threshold=gap_threshold,
        trailing_zero_min_days=trailing_zero_min_days,
    )

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))
    total = len(combos)
    print(
        f"\nGrid Search CV: {total} combinations x {n_windows} folds (horizon={horizon})"
    )

    gs_group = f"{pattern}_gs"  # tag to group all runs from this search
    results: list[dict] = []

    for idx, combo_values in enumerate(combos):
        combo: dict = dict(zip(keys, combo_values, strict=True))
        arima_kwargs = {k: v for k, v in combo.items() if k in _AUTOARIMA_KEYS}

        combo_label = "_".join(f"{k}{v}" for k, v in combo.items())
        run_name = f"{gs_group}_{idx + 1:03d}_{combo_label}"
        print(f"  [{idx + 1}/{total}] {combo_label} ...", end=" ", flush=True)

        _run_id = _client.create_run(
            experiment_id=_exp.experiment_id,
            run_name=run_name,
        ).info.run_id

        try:
            sf = StatsForecast(
                models=[AutoARIMA(season_length=season_length, **arima_kwargs)],
                freq=freq,
                n_jobs=1,
            )
            cv_df = sf.cross_validation(
                df=ts,
                h=horizon,
                step_size=step_size,
                n_windows=n_windows,
            )

            # MAE per fold → mean & std
            mae_per_fold = (
                cv_df.groupby("cutoff")
                .apply(
                    lambda g: mean_absolute_error(g["y"].values, g["AutoARIMA"].values),
                    include_groups=False,
                )
                .rename("mae")
            )
            cv_mae_mean = float(mae_per_fold.mean())
            cv_mae_std = float(mae_per_fold.std())
            print(f"MAE={cv_mae_mean:.3f} ± {cv_mae_std:.3f}")

            # Log params
            base_params = {
                "pattern": pattern,
                "model_type": "sarima",
                "freq": freq,
                "store": store,
                "item": item,
                "season_length": season_length,
                "cv_horizon": horizon,
                "cv_n_windows": n_windows,
                "cv_step_size": step_size,
                "cv_group": gs_group,
                **{f"mp_{_safe_param_key(k)}": v for k, v in combo.items()},
            }
            for key, val in base_params.items():
                _client.log_param(_run_id, key, str(val))
            _client.log_metric(_run_id, "cv_mae_mean", cv_mae_mean)
            _client.log_metric(_run_id, "cv_mae_std", cv_mae_std)
            for fold_idx, mae_val in enumerate(mae_per_fold):
                _client.log_metric(_run_id, "cv_mae_fold", mae_val, step=fold_idx)
            _client.set_tag(_run_id, "cv", "grid_search")
            _client.set_terminated(_run_id, "FINISHED")

            results.append(
                {
                    **combo,
                    "cv_mae_mean": cv_mae_mean,
                    "cv_mae_std": cv_mae_std,
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

    # Tag the best run
    best_run_id = results_df.loc[0, "run_id"]
    _client.set_tag(best_run_id, "best_in_group", gs_group)
    print(
        f"\nBest combo: {results_df.loc[0, list(keys)].to_dict()}  MAE={results_df.loc[0, 'cv_mae_mean']:.3f}"
    )

    return results_df
