from __future__ import annotations

from pathlib import Path

import pandas as pd

from Favorita_TSA.features.holidays import (
    HolidayConfig,
    build_store_calendar_daily,
    build_store_calendar_weekly_from_daily,
    load_holidays_events,
    load_stores,
)
from Favorita_TSA.utils.data_loader import df_to_parquet
from Favorita_TSA.utils.dataset import PreDataset

# Paths
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def _preprocessed_path(name: PreDataset) -> Path:
    return _project_root() / f"data/processed/preprocessed/{name.value}.parquet"

def _save(df: pd.DataFrame, rel_path: str) -> None:
    p = _project_root() / rel_path
    p.parent.mkdir(parents=True, exist_ok=True)
    df_to_parquet(df, str(p))

def enrich_store_item_daily(
    df_store_item_daily: pd.DataFrame,
    store_calendar_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds:
        is_holiday_or_event
        pre_holiday
        post_holiday
    Join keys: (store_nbr, date)
    """

    d = df_store_item_daily.copy()

    if "date" not in d.columns:
        raise ValueError("STORE_ITEM_DAILY must contain column 'date'")

    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.normalize()
    d = d.dropna(subset=["date"])

    cal = store_calendar_daily.copy()
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce").dt.normalize()
    cal = cal.dropna(subset=["date"])

    out = d.merge(
        cal[
            ["store_nbr", "date", "is_holiday_or_event", "pre_holiday", "post_holiday"]
        ],
        on=["store_nbr", "date"],
        how="left",
    )
    for c in ("is_holiday_or_event", "pre_holiday", "post_holiday"):
        if c not in out.columns:
            out[c] = False
        out[c] = out[c].fillna(False).astype(bool)

    return out

def enrich_store_item_weekly(
    df_store_item_weekly: pd.DataFrame,
    store_calendar_weekly: pd.DataFrame,
    week_start_col: str = "week_start",
) -> pd.DataFrame:
    """
    Adds:
        is_holiday_or_event
        pre_holiday
        post_holiday
    Join keys: (store_nbr, week_start)
    """

    w = df_store_item_weekly.copy()

    if week_start_col not in w.columns:
        raise ValueError(f"STORE_ITEM_WEEKLY must contain '{week_start_col}'")

    w[week_start_col] = pd.to_datetime(
        w[week_start_col], errors="coerce"
    ).dt.normalize()
    w = w.dropna(subset=[week_start_col])

    cal = store_calendar_weekly.copy()
    cal[week_start_col] = pd.to_datetime(
        cal[week_start_col], errors="coerce"
    ).dt.normalize()
    cal = cal.dropna(subset=[week_start_col])

    out = w.merge(
        cal[
            [
                "store_nbr",
                week_start_col,
                "is_holiday_or_event",
                "pre_holiday",
                "post_holiday",
            ]
        ],
        on=["store_nbr", week_start_col],
        how="left",
    )

    for c in ("is_holiday_or_event", "pre_holiday", "post_holiday"):
        if c not in out.columns:
            out[c] = False
        out[c] = out[c].fillna(False).astype(bool)

    return out

def save_holiday_enriched_tables(scope_days: int = 5) -> None:
    """
    Creates exactly two parquet files:

        store_item_daily_holiday.parquet
        store_item_weekly_holiday.parquet

    Columns added:
        is_holiday_or_event
        pre_holiday
        post_holiday
    """
    cfg = HolidayConfig()
    # Load base data
    df_stores = load_stores(cfg)
    df_holidays = load_holidays_events(cfg)

    df_store_item_daily = pd.read_parquet(_preprocessed_path(PreDataset.STORE_ITEM_DAILY))
    df_store_item_weekly = pd.read_parquet(_preprocessed_path(PreDataset.STORE_ITEM_WEEKLY))

    # Build calendars with scope
    store_cal_daily = build_store_calendar_daily(df_stores=df_stores,df_holidays_events=df_holidays,cfg=cfg,scope_days=scope_days,)
    store_cal_weekly = build_store_calendar_weekly_from_daily(store_calendar_daily=store_cal_daily,cfg=cfg,week_start_col="week_start",)

    # Enrich tables
    daily_out = enrich_store_item_daily(df_store_item_daily,store_cal_daily,)
    weekly_out = enrich_store_item_weekly(df_store_item_weekly,store_cal_weekly,week_start_col="week_start",)
    # Save
    _save(daily_out,"data/processed/preprocessed/store_item_daily_holiday.parquet",)
    _save(weekly_out,"data/processed/preprocessed/store_item_weekly_holiday.parquet",)
    print("✅ Holiday parquet tables saved:")
    print("   - store_item_daily_holiday.parquet")
    print("   - store_item_weekly_holiday.parquet")
