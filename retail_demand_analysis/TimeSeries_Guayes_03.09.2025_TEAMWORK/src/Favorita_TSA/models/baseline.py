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
    """
    from Favorita_TSA.utils.mlflow_utils import setup_mlflow

    print(f"\n{'=' * 70}\n{pattern.upper()} | {model_type.upper()}\n{'=' * 70}")

    experiment_name = mlflow_experiment or "favorita_baseline_store_item"
    setup_mlflow(experiment_name)

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
            df, store, item, freq=freq,
            gap_threshold=gap_threshold,
            trailing_zero_min_days=trailing_zero_min_days,
        )

        # 2. Split
        train, test = train_test_split(ts, test_weeks=test_weeks)

        # 3. Primary model (Theta oder SARIMA)
        if model_type == "theta":
            theta_kwargs = {k: v for k, v in model_params.items() if k in _THETA_KEYS}
            model_primary = StatsForecast(
                models=[Theta(season_length=season_length, **theta_kwargs)],
                freq=freq, n_jobs=1,
            )
            primary_col = "Theta"
        else:
            arima_kwargs = {k: v for k, v in model_params.items() if k in _AUTOARIMA_KEYS}
            model_primary = StatsForecast(
                models=[AutoARIMA(season_length=season_length, **arima_kwargs)],
                freq=freq, n_jobs=1,
            )
            primary_col = "AutoARIMA"

        model_primary.fit(train)

        # 4. Forecast & Evaluation
        horizon = len(test)
        forecasts_primary = model_primary.predict(h=horizon).reset_index()
        test_primary = test.copy()
        test_primary[primary_col] = forecasts_primary[primary_col].values

        actuals = test_primary["y"].values
        preds_primary = test_primary[primary_col].values
        mask = ~np.isnan(actuals) & ~np.isnan(preds_primary)

        mae_primary = mean_absolute_error(actuals[mask], preds_primary[mask])
        r2_primary = r2_score(actuals[mask], preds_primary[mask])

        # 5. Naive benchmark
        model_naive = StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq=freq, n_jobs=1)
        model_naive.fit(train)
        forecasts_naive = model_naive.predict(h=horizon).reset_index()
        test_naive = test.copy()
        test_naive["SeasonalNaive"] = forecasts_naive["SeasonalNaive"].values
        mae_naive = mean_absolute_error(actuals[mask], test_naive["SeasonalNaive"].values[mask])
        r2_naive = r2_score(actuals[mask], test_naive["SeasonalNaive"].values[mask])

        improvement = 0.0 if mae_naive == 0 else (mae_naive - mae_primary) / mae_naive * 100

        # 6. Plotting (Subplots für Übersicht)
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.12)
        fig.add_trace(go.Scatter(x=train["ds"], y=train["y"], name="Train"), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_primary["ds"], y=test_primary["y"], name="Actual"), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_primary["ds"], y=test_primary[primary_col], name=f"{model_type.upper()}", line={"dash": "dash"}), row=1, col=1)
        # Test-Zoom-Ansicht in Reihe 2
        fig.add_trace(go.Scatter(x=test_primary["ds"], y=test_primary["y"], showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=test_primary["ds"], y=test_primary[primary_col], showlegend=False, line={"dash": "dash"}), row=2, col=1)
        fig.update_layout(height=800)

        # 7. MLflow Logging
        params = {
            "pattern": pattern, "model_type": model_type, "store": store, "item": item,
            **{f"mp_{_safe_param_key(k)}": v for k, v in model_params.items()},
        }
        metrics = {
            "mae_primary": float(mae_primary),
            "mae_naive": float(mae_naive),
            "r2_primary": float(r2_primary),
            "improvement_pct": float(improvement),
            "test_weeks": test_weeks,
            "test_size": len(test),
        }

        mlflow_metrics = {k: v for k, v in metrics.items() if isinstance(v, float)}
        mlflow_params = {**params, **{k: v for k, v in metrics.items() if not isinstance(v, float)}}
        for k, v in mlflow_params.items(): _client.log_param(_run_id, k, str(v))
        for k, v in mlflow_metrics.items(): _client.log_metric(_run_id, k, v)

        # Artefakte (Plots & Tabellen) speichern
        with tempfile.TemporaryDirectory() as _tmp:
            _fig_path = Path(_tmp) / "forecast.html"
            fig.write_html(str(_fig_path))
            _client.log_artifact(_run_id, str(_fig_path), artifact_path="plots")

        _client.set_terminated(_run_id, "FINISHED")

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
