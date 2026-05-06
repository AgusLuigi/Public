# src/Favorita_TSA/features/holidays.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from Favorita_TSA.utils.data_loader import parquet_loader

# =====================================================
# Config
# =====================================================


@dataclass(frozen=True)
class HolidayConfig:
    # stores
    stores_store_col: str = "store_nbr"
    stores_state_col: str = "state"

    # holidays_events
    hol_date_col: str = "date"
    hol_locale_name_col: str = "locale_name"
    hol_type_col: str = "type"
    hol_transferred_col: str = "transferred"

    # selection
    country_locale: str = "Ecuador"
    ignore_types: tuple[str, ...] = ("Work Day",)
    ignore_transferred: bool = True

    # output columns (FIXED names expected downstream)
    is_holiday_col: str = "is_holiday_or_event"
    pre_holiday_col: str = "pre_holiday"
    post_holiday_col: str = "post_holiday"
    holiday_scope_col: str = "holiday_scope"  # optional (holiday/pre/post/none)


# =====================================================
# Path helpers (robust in notebooks/streamlit)
# =====================================================


def _project_root() -> Path:
    # file: src/Favorita_TSA/features/holidays.py -> parents[3] = repo root
    return Path(__file__).resolve().parents[3]


def _safe_parquet_loader(dataset, fallback_rel: str) -> pd.DataFrame:
    try:
        return parquet_loader(dataset)  # type: ignore[arg-type]
    except FileNotFoundError:
        return pd.read_parquet(_project_root() / fallback_rel)


# =====================================================
# Loaders (use your existing Dataset enum)
# =====================================================


def load_stores(cfg: HolidayConfig):
    # Dataset.STORES expected to exist in your project
    from Favorita_TSA.utils.dataset import Dataset  # local import to avoid cycles

    df = _safe_parquet_loader(Dataset.STORES, "data/processed/stores.parquet")
    missing = {cfg.stores_store_col, cfg.stores_state_col} - set(df.columns)
    if missing:
        raise ValueError(f"stores.parquet missing columns: {sorted(missing)}")
    out = df[[cfg.stores_store_col, cfg.stores_state_col]].copy()
    out[cfg.stores_store_col] = out[cfg.stores_store_col].astype(int)
    out[cfg.stores_state_col] = out[cfg.stores_state_col].astype(str)
    return out


def load_holidays_events(cfg: HolidayConfig):
    from Favorita_TSA.utils.dataset import Dataset  # local import

    df = _safe_parquet_loader(
        Dataset.HOLIDAYS_EVENTS, "data/processed/holidays_events.parquet"
    )
    missing = {cfg.hol_date_col, cfg.hol_locale_name_col} - set(df.columns)
    if missing:
        raise ValueError(f"holidays_events.parquet missing columns: {sorted(missing)}")
    return df.copy()


# =====================================================
# Core cleaning + selection
# =====================================================


def _clean_holidays_events(df: pd.DataFrame, cfg: HolidayConfig) -> pd.DataFrame:
    out = df.copy()
    out[cfg.hol_date_col] = pd.to_datetime(
        out[cfg.hol_date_col], errors="coerce"
    ).dt.normalize()
    out = out.dropna(subset=[cfg.hol_date_col])

    # ignore types like Work Day
    if cfg.hol_type_col in out.columns:
        out = out[~out[cfg.hol_type_col].isin(cfg.ignore_types)]

    # ignore transferred holidays (transferred=True means holiday moved; original day is work)
    if cfg.ignore_transferred and cfg.hol_transferred_col in out.columns:
        out = out[~out[cfg.hol_transferred_col].fillna(False).astype(bool)]

    return out


def _holiday_dates_for_state(
    holidays_clean: pd.DataFrame,
    state: str,
    cfg: HolidayConfig,
) -> pd.DatetimeIndex:
    # country + state scope
    allowed = {cfg.country_locale, str(state)}
    h = holidays_clean[holidays_clean[cfg.hol_locale_name_col].isin(allowed)]
    return pd.DatetimeIndex(h[cfg.hol_date_col].unique()).sort_values()


# =====================================================
# Scope expansion (pre/post days)
# =====================================================


def _build_scoped_calendar_for_store(
    store_nbr: int,
    holiday_dates: pd.DatetimeIndex,
    scope_days: int,
    cfg: HolidayConfig,
) -> pd.DataFrame:
    """
    Produces rows for one store:
      store_nbr, date, is_holiday_or_event, pre_holiday, post_holiday, holiday_scope
    """
    if len(holiday_dates) == 0:
        return pd.DataFrame(
            columns=[
                cfg.stores_store_col,
                cfg.hol_date_col,
                cfg.is_holiday_col,
                cfg.pre_holiday_col,
                cfg.post_holiday_col,
                cfg.holiday_scope_col,
            ]
        )

    # exact holiday dates
    hol = pd.DataFrame(
        {
            cfg.stores_store_col: store_nbr,
            cfg.hol_date_col: holiday_dates,
            cfg.is_holiday_col: True,
            cfg.pre_holiday_col: False,
            cfg.post_holiday_col: False,
            cfg.holiday_scope_col: "holiday",
        }
    )

    if scope_days <= 0:
        return hol

    # pre/post ranges per holiday
    pre_rows: list[pd.DataFrame] = []
    post_rows: list[pd.DataFrame] = []

    for d in holiday_dates:
        pre_rng = pd.date_range(
            d - pd.Timedelta(days=scope_days), d - pd.Timedelta(days=1), freq="D"
        )
        post_rng = pd.date_range(
            d + pd.Timedelta(days=1), d + pd.Timedelta(days=scope_days), freq="D"
        )

        if len(pre_rng) > 0:
            pre_rows.append(
                pd.DataFrame(
                    {
                        cfg.stores_store_col: store_nbr,
                        cfg.hol_date_col: pre_rng,
                        cfg.is_holiday_col: False,
                        cfg.pre_holiday_col: True,
                        cfg.post_holiday_col: False,
                        cfg.holiday_scope_col: "pre",
                    }
                )
            )

        if len(post_rng) > 0:
            post_rows.append(
                pd.DataFrame(
                    {
                        cfg.stores_store_col: store_nbr,
                        cfg.hol_date_col: post_rng,
                        cfg.is_holiday_col: False,
                        cfg.pre_holiday_col: False,
                        cfg.post_holiday_col: True,
                        cfg.holiday_scope_col: "post",
                    }
                )
            )

    out = pd.concat([hol, *pre_rows, *post_rows], ignore_index=True)
    out[cfg.hol_date_col] = pd.to_datetime(
        out[cfg.hol_date_col], errors="coerce"
    ).dt.normalize()
    out = out.dropna(subset=[cfg.hol_date_col])

    # If overlaps happen (e.g., scopes overlap), resolve priority:
    # holiday > pre/post. And if pre+post overlap, keep both flags True and scope="pre/post".
    out = out.groupby([cfg.stores_store_col, cfg.hol_date_col], as_index=False).agg(
        {
            cfg.is_holiday_col: "max",
            cfg.pre_holiday_col: "max",
            cfg.post_holiday_col: "max",
        }
    )

    def _scope_row(r):
        if r[cfg.is_holiday_col]:
            return "holiday"
        if r[cfg.pre_holiday_col] and r[cfg.post_holiday_col]:
            return "pre/post"
        if r[cfg.pre_holiday_col]:
            return "pre"
        if r[cfg.post_holiday_col]:
            return "post"
        return "none"

    out[cfg.holiday_scope_col] = out.apply(_scope_row, axis=1)

    # enforce bool
    for c in (cfg.is_holiday_col, cfg.pre_holiday_col, cfg.post_holiday_col):
        out[c] = out[c].fillna(False).astype(bool)

    return out


# =====================================================
# Public builders
# =====================================================


def build_store_calendar_daily(
    df_stores: pd.DataFrame,
    df_holidays_events: pd.DataFrame,
    cfg: HolidayConfig,
    scope_days: int = 5,
) -> pd.DataFrame:
    """
    Returns store-level daily calendar:
      store_nbr, date, is_holiday_or_event, pre_holiday, post_holiday, holiday_scope
    """
    stores = df_stores[[cfg.stores_store_col, cfg.stores_state_col]].copy()
    stores[cfg.stores_store_col] = stores[cfg.stores_store_col].astype(int)

    holidays_clean = _clean_holidays_events(df_holidays_events, cfg)

    parts: list[pd.DataFrame] = []
    for store_nbr, state in stores.itertuples(index=False):
        dates = _holiday_dates_for_state(holidays_clean, state=str(state), cfg=cfg)
        parts.append(
            _build_scoped_calendar_for_store(int(store_nbr), dates, scope_days, cfg)
        )

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(
            columns=[
                cfg.stores_store_col,
                cfg.hol_date_col,
                cfg.is_holiday_col,
                cfg.pre_holiday_col,
                cfg.post_holiday_col,
                cfg.holiday_scope_col,
            ]
        )

    out = out.drop_duplicates([cfg.stores_store_col, cfg.hol_date_col])
    out = out.sort_values([cfg.stores_store_col, cfg.hol_date_col])
    return out


def build_store_calendar_weekly_from_daily(
    store_calendar_daily: pd.DataFrame,
    cfg: HolidayConfig,
    week_start_col: str = "week_start",
) -> pd.DataFrame:
    """
    Aggregates the store daily calendar into weekly calendar.
    Weekly flags are True if ANY day in that week has that flag.
    """
    cal = store_calendar_daily.copy()
    if cfg.hol_date_col not in cal.columns:
        raise ValueError(f"store_calendar_daily missing '{cfg.hol_date_col}'")

    cal[cfg.hol_date_col] = pd.to_datetime(
        cal[cfg.hol_date_col], errors="coerce"
    ).dt.normalize()
    cal = cal.dropna(subset=[cfg.hol_date_col])

    # week start = Monday
    cal[week_start_col] = cal[cfg.hol_date_col] - pd.to_timedelta(
        cal[cfg.hol_date_col].dt.weekday, unit="D"
    )

    out = cal.groupby([cfg.stores_store_col, week_start_col], as_index=False).agg(
        {
            cfg.is_holiday_col: "any",
            cfg.pre_holiday_col: "any",
            cfg.post_holiday_col: "any",
        }
    )

    def _scope_row(r):
        if r[cfg.is_holiday_col]:
            return "holiday"
        if r[cfg.pre_holiday_col] and r[cfg.post_holiday_col]:
            return "pre/post"
        if r[cfg.pre_holiday_col]:
            return "pre"
        if r[cfg.post_holiday_col]:
            return "post"
        return "none"

    out[cfg.holiday_scope_col] = out.apply(_scope_row, axis=1)

    for c in (cfg.is_holiday_col, cfg.pre_holiday_col, cfg.post_holiday_col):
        out[c] = out[c].fillna(False).astype(bool)

    out[week_start_col] = pd.to_datetime(
        out[week_start_col], errors="coerce"
    ).dt.normalize()
    out = out.dropna(subset=[week_start_col])
    out = out.sort_values([cfg.stores_store_col, week_start_col])

    return out
