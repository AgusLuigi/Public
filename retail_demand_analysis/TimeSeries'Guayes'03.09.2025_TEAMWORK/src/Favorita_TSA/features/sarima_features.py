# src/Favorita_TSA/features/sarima_features.py
"""
Feature engineering pipeline for SARIMAX models.

Builds two enriched segment parquets (smooth_daily, erratic_daily) starting
from the exact same fact-table slices used by the baseline SARIMA models,
then adds exogenous features:

  Exogenous group        Features
  ─────────────────────  ──────────────────────────────────────────────────────
  Holidays               is_holiday_or_event, pre_holiday, post_holiday
  Oil price              oil_price, oil_price_ma7, oil_price_ma28,
                         oil_price_pct_change
  Calendar               is_weekend, is_payday, is_month_start, is_month_end,
                         days_to_next_holiday, days_since_last_holiday
  Transactions           transactions, transactions_ma7, transactions_z_score
  Promotion (extended)   promo_streak, promo_rate_7d
                         (onpromotion already in fact table)
  Store metadata         store_type, store_cluster
  Item metadata          family, perishable

Usage
─────
  from Favorita_TSA.features.sarima_features import build_sarimax_segments

  results = build_sarimax_segments()
  smooth  = results["smooth_daily"]
  erratic = results["erratic_daily"]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Favorita_TSA.features.holidays import (
    HolidayConfig,
    build_store_calendar_daily,
    load_holidays_events,
    load_stores,
)
from Favorita_TSA.models.data_preparation import build_dataframes
from Favorita_TSA.utils.paths import PREPROCESSED_DIR, PROCESSED_DIR

_KEY_COLS = ["store_nbr", "item_nbr"]


# ─── Oil features ─────────────────────────────────────────────────────────────


def load_oil_features(oil_path: Path | None = None) -> pd.DataFrame:
    """
    Load daily oil prices, forward-fill weekend/holiday gaps, and add
    rolling averages and a percentage-change signal.

    Returns a DataFrame with columns:
        date, oil_price, oil_price_ma7, oil_price_ma28, oil_price_pct_change
    """
    if oil_path is None:
        oil_path = PROCESSED_DIR / "oil.parquet"

    oil = pd.read_parquet(oil_path).rename(columns={"dcoilwtico": "oil_price"})
    oil["date"] = pd.to_datetime(oil["date"])

    # Reindex to fill all calendar days (oil data has no weekend entries)
    oil = oil.set_index("date").sort_index()
    full_range = pd.date_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full_range)
    oil.index.name = "date"

    oil["oil_price"] = oil["oil_price"].ffill().bfill()
    oil["oil_price_ma7"] = oil["oil_price"].rolling(7, min_periods=1).mean()
    oil["oil_price_ma28"] = oil["oil_price"].rolling(28, min_periods=1).mean()
    oil["oil_price_pct_change"] = oil["oil_price"].pct_change().fillna(0)

    return oil.reset_index()


# ─── Transaction features ─────────────────────────────────────────────────────


def load_transaction_features(tx_path: Path | None = None) -> pd.DataFrame:
    """
    Load daily store transaction counts, add a 7-day rolling mean and a
    store-normalised z-score (removes store-size effect).

    Returns a DataFrame with columns:
        date, store_nbr, transactions, transactions_ma7, transactions_z_score
    """
    if tx_path is None:
        tx_path = PROCESSED_DIR / "transactions.parquet"

    tx = pd.read_parquet(tx_path)
    tx["date"] = pd.to_datetime(tx["date"])
    tx = tx.sort_values(["store_nbr", "date"])

    tx["transactions_ma7"] = tx.groupby("store_nbr")["transactions"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )

    # z-score per store: (value - store_mean) / store_std
    store_stats = tx.groupby("store_nbr")["transactions"].agg(["mean", "std"])
    tx = tx.merge(store_stats, on="store_nbr", how="left")
    tx["transactions_z_score"] = (tx["transactions"] - tx["mean"]) / tx["std"].clip(
        lower=1e-6
    )
    tx = tx.drop(columns=["mean", "std"])

    return tx


# ─── Holiday calendar ─────────────────────────────────────────────────────────


def load_store_holiday_calendar(scope_days: int = 5) -> pd.DataFrame:
    """
    Build the store-level daily holiday calendar using the existing
    holiday pipeline (respects local/national scopes, ignores Work Day
    and transferred holidays).

    Returns a DataFrame with columns:
        store_nbr, date, is_holiday_or_event, pre_holiday, post_holiday
    """
    cfg = HolidayConfig()
    df_stores = load_stores(cfg)
    df_holidays = load_holidays_events(cfg)
    cal = build_store_calendar_daily(df_stores, df_holidays, cfg, scope_days=scope_days)
    return cal[
        ["store_nbr", "date", "is_holiday_or_event", "pre_holiday", "post_holiday"]
    ]


# ─── Calendar features ────────────────────────────────────────────────────────


def _national_holiday_dates_int() -> np.ndarray:
    """Ecuador national holiday dates as sorted int64 (days since epoch)."""
    holidays = pd.read_parquet(PROCESSED_DIR / "holidays_events.parquet")
    national = holidays[
        (holidays["locale_name"] == "Ecuador")
        & (~holidays["transferred"].fillna(False))
        & (holidays["type"] != "Work Day")
    ]
    dates = pd.to_datetime(national["date"].unique())
    return np.sort(dates.values.astype("datetime64[D]").view(np.int64))


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-based exogenous features derived from the 'date' column:

        is_weekend            : Sa/So = 1
        is_payday             : 15th and last day of month (Ecuador payroll)
        is_month_start        : first 3 days of month
        is_month_end          : last 3 days of month
        days_to_next_holiday  : days until next national holiday (-1 if none)
        days_since_last_holiday: days since last national holiday (-1 if none)
    """
    holiday_ints = _national_holiday_dates_int()
    dates = pd.to_datetime(df["date"])

    df = df.copy()
    df["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(np.int8)
    df["is_payday"] = (
        dates.dt.day.isin([15]) | (dates.dt.day == dates.dt.days_in_month)
    ).astype(np.int8)
    df["is_month_start"] = (dates.dt.day <= 3).astype(np.int8)
    df["is_month_end"] = (dates.dt.day >= dates.dt.days_in_month - 2).astype(np.int8)

    # Vectorised holiday-distance computation via binary search
    dates_ints = dates.values.astype("datetime64[D]").view(np.int64)
    idx = np.searchsorted(holiday_ints, dates_ints)

    df["days_to_next_holiday"] = np.where(
        idx < len(holiday_ints),
        holiday_ints[np.clip(idx, 0, len(holiday_ints) - 1)] - dates_ints,
        -1,
    ).astype(np.int16)

    df["days_since_last_holiday"] = np.where(
        idx > 0,
        dates_ints - holiday_ints[np.clip(idx - 1, 0, len(holiday_ints) - 1)],
        -1,
    ).astype(np.int16)

    return df


# ─── Promotion features ───────────────────────────────────────────────────────


def _promo_streak(s: pd.Series) -> pd.Series:
    """
    Consecutive days in promotion for a single time series.
    Resets to 0 on non-promo days.

    E.g. [0,0,1,1,1,0,1,1] → [0,0,1,2,3,0,1,2]
    """
    s = s.fillna(False).astype(int)
    cumsum = s.cumsum()
    # At each zero, record the running cumsum value; forward-fill to get the
    # last cumsum at the most recent zero — subtracting gives the streak.
    cumsum_at_last_zero = cumsum.where(s == 0).ffill().fillna(0)
    return (cumsum - cumsum_at_last_zero).astype(np.int16)


def build_promotion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add promotion features per (store_nbr, item_nbr):

        promo_streak   : consecutive days currently in promotion
        promo_rate_7d  : fraction of the last 7 days with promotion (0-1)

    Requires columns: store_nbr, item_nbr, date, onpromotion.
    """
    df = df.copy().sort_values([*_KEY_COLS, "date"])
    grouped = df.groupby(_KEY_COLS)["onpromotion"]

    df["promo_rate_7d"] = grouped.transform(
        lambda s: s.astype(float).rolling(7, min_periods=1).mean()
    )
    df["promo_streak"] = grouped.transform(_promo_streak)

    return df


# ─── Store / Item metadata ────────────────────────────────────────────────────


def load_store_item_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load static store and item metadata for use as categorical covariates.

    Returns
    -------
    stores : DataFrame [store_nbr, store_type, store_cluster]
    items  : DataFrame [item_nbr, family, perishable]
    """
    stores = pd.read_parquet(
        PROCESSED_DIR / "stores.parquet", columns=["store_nbr", "type", "cluster"]
    ).rename(columns={"type": "store_type", "cluster": "store_cluster"})

    items = pd.read_parquet(
        PROCESSED_DIR / "items.parquet", columns=["item_nbr", "family", "perishable"]
    )
    return stores, items


# ─── Segment enrichment ───────────────────────────────────────────────────────


def enrich_segment(
    segment_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    tx_df: pd.DataFrame,
    store_holiday_cal: pd.DataFrame,
    stores_meta: pd.DataFrame,
    items_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich one smooth/erratic daily segment with all exogenous features.

    The returned DataFrame is schema-compatible with the baseline input
    (same ds/y/unique_id columns) plus additional exogenous columns that
    can be passed as X to SARIMAX.
    """
    df = segment_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 1. Holiday flags (store-level calendar, merge on store_nbr + date)
    df = df.merge(store_holiday_cal, on=["store_nbr", "date"], how="left")
    for col in ["is_holiday_or_event", "pre_holiday", "post_holiday"]:
        df[col] = df[col].astype(object).fillna(False).astype(bool)

    # 2. Oil features (global, merge on date)
    df = df.merge(oil_df, on="date", how="left")

    # 3. Transaction features (store-level, merge on store_nbr + date)
    df = df.merge(tx_df, on=["store_nbr", "date"], how="left")
    df["transactions"] = df["transactions"].fillna(0).astype(np.int32)
    df["transactions_ma7"] = df["transactions_ma7"].fillna(0)
    df["transactions_z_score"] = df["transactions_z_score"].fillna(0)

    # 4. Calendar features (derived from date column)
    df = build_calendar_features(df)

    # 5. Promotion features (per store-item group, derived from onpromotion)
    df = build_promotion_features(df)

    # 6. Store metadata (static, merge on store_nbr)
    df = df.merge(stores_meta, on="store_nbr", how="left")

    # 7. Item metadata (static, merge on item_nbr)
    df = df.merge(items_meta, on="item_nbr", how="left")

    return df


# ─── Weekly aggregation ───────────────────────────────────────────────────────

# Aggregationslogik pro Feature-Gruppe
_WEEKLY_AGG: dict[str, str] = {
    "unit_sales": "sum",
    "onpromotion": "max",
    "is_holiday_or_event": "max",
    "pre_holiday": "max",
    "post_holiday": "max",
    "oil_price": "mean",
    "oil_price_ma7": "mean",
    "oil_price_ma28": "mean",
    "oil_price_pct_change": "mean",
    "transactions": "sum",
    "transactions_ma7": "mean",
    "transactions_z_score": "mean",
    "is_payday": "max",
    "is_month_start": "max",
    "is_month_end": "max",
    "days_to_next_holiday": "min",
    "days_since_last_holiday": "min",
    "promo_streak": "max",
    "promo_rate_7d": "mean",
    "store_type": "first",
    "store_cluster": "first",
    "family": "first",
    "perishable": "first",
}
# is_weekend wird weggelassen (auf Wochenebene sinnlos)


def aggregate_features_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert einen tagesgenauen (enriched) Store-Item-DataFrame auf Wochenebene.

    Die Woche beginnt am Montag (week_start).
    Jede Feature-Gruppe wird mit der semantisch passenden Aggregation
    zusammengefasst (sum / mean / max / min / first).
    ``is_weekend`` wird weggelassen, da es auf Wochenebene bedeutungslos ist.

    Parameters
    ----------
    df : DataFrame
        Tagesgenaue Zeitreihe mit mindestens den Spalten
        store_nbr, item_nbr, date, unit_sales.

    Returns
    -------
    DataFrame mit Spalte ``week_start`` statt ``date``.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")

    present_cols = {c: agg for c, agg in _WEEKLY_AGG.items() if c in df.columns}
    present_cols["week_start"] = (
        "first"  # Platzhalter — wird als Gruppierschlüssel genutzt
    )

    agg_spec = {c: agg for c, agg in present_cols.items() if c != "week_start"}

    weekly = (
        df.groupby([*_KEY_COLS, "week_start"], sort=True).agg(agg_spec).reset_index()
    )
    return weekly


# ─── Segment-Loader für SARIMAX (täglich + wöchentlich) ──────────────────────

_PATTERN_TO_SEGMENT: dict[str, str] = {
    "daily_smooth": "smooth_daily",
    "daily_erratic": "erratic_daily",
    "weekly_smooth": "smooth_weekly",
    "weekly_erratic": "erratic_weekly",
}


def load_sarimax_segment(pattern: str) -> pd.DataFrame:
    """
    Gibt einen mit exogenen Features angereicherten DataFrame für das
    gewünschte Demand-Pattern zurück.

    Für tägliche Patterns (daily_smooth / daily_erratic):
        Wird direkt aus ``build_sarimax_segments()`` geladen.

    Für wöchentliche Patterns (weekly_smooth / weekly_erratic):
        1. Das entsprechende Segment aus ``build_dataframes()`` laden
           (enthält nur store_nbr, item_nbr, week_start, unit_sales, …).
        2. Die tägliche Fact-Table für alle Store-Item-Paare laden.
        3. Exogene Features anreichern.
        4. Auf Wochenebene aggregieren via ``aggregate_features_to_weekly()``.

    Parameters
    ----------
    pattern : str
        Eines von: "daily_smooth", "daily_erratic",
                   "weekly_smooth", "weekly_erratic".

    Returns
    -------
    DataFrame bereit für SARIMAX (ds/y/unique_id + exogene Spalten).
    """
    if pattern not in _PATTERN_TO_SEGMENT:
        raise ValueError(
            f"Unbekanntes Pattern '{pattern}'. " f"Erlaubt: {list(_PATTERN_TO_SEGMENT)}"
        )

    if pattern.startswith("daily"):
        segment_key = _PATTERN_TO_SEGMENT[pattern]
        segments = build_sarimax_segments()
        return segments[segment_key]

    # ── Wöchentlich: tägliche Basis laden, anreichern, aggregieren ───────────
    # 1. Wöchentliches Segment → nur um Store-Item-Paare zu ermitteln
    segment_key = _PATTERN_TO_SEGMENT[pattern]
    base_dfs = build_dataframes()
    weekly_seg = base_dfs[segment_key]
    pairs = weekly_seg[_KEY_COLS].drop_duplicates()

    # 2. Tägliche Fact-Table für diese Paare laden
    fact = pd.read_parquet(PREPROCESSED_DIR / "fact_table.parquet")
    fact["date"] = pd.to_datetime(fact["date"])
    fact = fact.merge(pairs, on=_KEY_COLS, how="inner")

    # 3. Exogene Features laden und anreichern
    oil_df = load_oil_features()
    tx_df = load_transaction_features()
    store_holiday_cal = load_store_holiday_calendar()
    stores_meta, items_meta = load_store_item_metadata()

    enriched = enrich_segment(
        fact, oil_df, tx_df, store_holiday_cal, stores_meta, items_meta
    )

    # 4. Auf Wochenebene aggregieren
    return aggregate_features_to_weekly(enriched)


# ─── Main pipeline ────────────────────────────────────────────────────────────


def build_sarimax_segments() -> dict[str, pd.DataFrame]:
    """
    Build smooth_daily and erratic_daily segments enriched with all
    exogenous features.

    Both segments start from the exact same fact-table slice as the
    baseline SARIMA models (same store-item pairs, same date range,
    same unit_sales target — no scaling or transformation applied).

    Returns
    -------
    dict with keys:
        "smooth_daily"  - Smooth pattern daily segment + exogenous features
        "erratic_daily" - Erratic pattern daily segment + exogenous features
    """
    print("Building base segments from fact table …")
    base_dfs = build_dataframes()

    print("Loading exogenous data …")
    oil_df = load_oil_features()
    tx_df = load_transaction_features()
    store_holiday_cal = load_store_holiday_calendar()
    stores_meta, items_meta = load_store_item_metadata()

    results: dict[str, pd.DataFrame] = {}
    for key in ["smooth_daily", "erratic_daily"]:
        print(f"Enriching {key} …")
        results[key] = enrich_segment(
            base_dfs[key],
            oil_df,
            tx_df,
            store_holiday_cal,
            stores_meta,
            items_meta,
        )
        n_pairs = results[key][_KEY_COLS].drop_duplicates().shape[0]
        n_cols = len(results[key].columns)
        print(
            f"  {key}: {len(results[key]):,} rows | "
            f"{n_pairs:,} store-item pairs | {n_cols} columns"
        )

    return results
