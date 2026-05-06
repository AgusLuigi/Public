"""
data_preparation.py

Splits the Favorita fact table into four DataFrames based on the
Croston demand-pattern classification (Syntetos-Boylan matrix).

The classification comes from two forecastability matrices:
  - store_item_daily_metrics   → daily patterns  (Smooth / Erratic / Intermittent / Lumpy)
  - store_item_weekly_metrics  → weekly patterns (Smooth / Erratic / Intermittent / Lumpy)

We extract the (store_nbr, item_nbr) keys for each pattern and use them
to filter the fact table.  All four output DataFrames share the same schema
as the fact table.

Output DataFrames
─────────────────
  df_smooth_daily    - items classified as Smooth  in the daily  matrix
  df_erratic_daily   - items classified as Erratic in the daily  matrix
  df_smooth_weekly   - items classified as Smooth  in the weekly matrix
  df_erratic_weekly  - items classified as Erratic in the weekly matrix

Quick start
───────────
  from Favorita_TSA.models.data_preparation import build_dataframes

  dfs = build_dataframes()
  df_smooth_daily   = dfs["smooth_daily"]
  df_erratic_daily  = dfs["erratic_daily"]
  df_smooth_weekly  = dfs["smooth_weekly"]
  df_erratic_weekly = dfs["erratic_weekly"]
"""

from pathlib import Path

import pandas as pd

from Favorita_TSA.utils.paths import METRICS_DIR
from Favorita_TSA.utils.preprocess_data import load_fact_table

# ─────────────────────────────────────────────────────────────────────────────
# Paths to the forecastability matrices
# ─────────────────────────────────────────────────────────────────────────────

DAILY_METRICS_PATH = METRICS_DIR / "store_item_daily_metrics.parquet"
WEEKLY_METRICS_PATH = METRICS_DIR / "store_item_weekly_metrics.parquet"

# Columns that identify a unique store-item combination
_KEY_COLS = ["store_nbr", "item_nbr"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _get_keys(metrics_path: Path, pattern: str) -> pd.DataFrame:
    """
    Read the metrics parquet and return the (store_nbr, item_nbr) pairs
    that match *pattern* (e.g. 'Smooth' or 'Erratic').
    Only the three needed columns are loaded to keep memory usage low.
    """
    metrics = pd.read_parquet(metrics_path, columns=[*_KEY_COLS, "pattern"])
    keys = metrics.loc[metrics["pattern"] == pattern, _KEY_COLS].drop_duplicates()
    return keys.reset_index(drop=True)


def _filter_fact_table(fact_table: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the rows in *fact_table* whose (store_nbr, item_nbr)
    appear in *keys*.  Returns a clean, re-indexed DataFrame.
    """
    return fact_table.merge(keys, on=_KEY_COLS, how="inner").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_dataframes(
    fact_table: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build the four segment DataFrames from the fact table.

    The fact table is loaded from disk when not provided.
    Each returned DataFrame has the same schema as the fact table:
        store_nbr, item_nbr, date, unit_sales, onpromotion,
        year, dow, year_iso, week, week_start, month

    Returns
    -------
    dict with keys:
        "smooth_daily"   - items classified Smooth  in the daily  matrix
        "erratic_daily"  - items classified Erratic in the daily  matrix
        "smooth_weekly"  - items classified Smooth  in the weekly matrix
        "erratic_weekly" - items classified Erratic in the weekly matrix
    """
    # --- load data -----------------------------------------------------------
    print("Loading fact table …")
    if fact_table is None:
        fact_table = load_fact_table()

    print("Loading forecastability matrices …")
    daily_metrics = pd.read_parquet(DAILY_METRICS_PATH, columns=[*_KEY_COLS, "pattern"])
    weekly_metrics = pd.read_parquet(
        WEEKLY_METRICS_PATH, columns=[*_KEY_COLS, "pattern"]
    )

    # --- extract (store_nbr, item_nbr) keys per pattern ----------------------
    smooth_daily_keys = daily_metrics.loc[
        daily_metrics["pattern"] == "Smooth", _KEY_COLS
    ].drop_duplicates()
    erratic_daily_keys = daily_metrics.loc[
        daily_metrics["pattern"] == "Erratic", _KEY_COLS
    ].drop_duplicates()
    smooth_weekly_keys = weekly_metrics.loc[
        weekly_metrics["pattern"] == "Smooth", _KEY_COLS
    ].drop_duplicates()
    erratic_weekly_keys = weekly_metrics.loc[
        weekly_metrics["pattern"] == "Erratic", _KEY_COLS
    ].drop_duplicates()

    # --- filter fact table for each segment ----------------------------------
    print("Building DataFrames …")
    df_smooth_daily = _filter_fact_table(fact_table, smooth_daily_keys)
    df_erratic_daily = _filter_fact_table(fact_table, erratic_daily_keys)
    df_smooth_weekly = _filter_fact_table(fact_table, smooth_weekly_keys)
    df_erratic_weekly = _filter_fact_table(fact_table, erratic_weekly_keys)

    # --- summary -------------------------------------------------------------
    results = {
        "smooth_daily": df_smooth_daily,
        "erratic_daily": df_erratic_daily,
        "smooth_weekly": df_smooth_weekly,
        "erratic_weekly": df_erratic_weekly,
    }
    for name, df in results.items():
        n_pairs = df[_KEY_COLS].drop_duplicates().shape[0]
        print(f"  {name:<18}  {n_pairs:>7,} store-item pairs   {len(df):>12,} rows")

    return results
