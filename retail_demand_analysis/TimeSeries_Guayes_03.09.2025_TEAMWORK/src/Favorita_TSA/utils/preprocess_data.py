from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from Favorita_TSA.utils.data_loader import df_to_parquet, parquet_loader, parquet_save
from Favorita_TSA.utils.dataset import Dataset, PreDataset
from Favorita_TSA.utils.paths import PREPROCESSED_DIR, METRICS_DIR


def create_fact_table():
    df = parquet_loader(Dataset.TRAIN).copy()

    df["date"] = pd.to_datetime(df["date"])

    # -------------------------
    # Daily
    # -------------------------
    df["year"] = df["date"].dt.year
    df["dow"] = df["date"].dt.dayofweek

    # -------------------------
    # Weekly (ISO clean)
    # -------------------------
    iso = df["date"].dt.isocalendar()

    df["year_iso"] = iso.year
    df["week"] = iso.week.astype(int)

    # Montag als Week Start (Timestamp)
    df["week_start"] = df["date"] - pd.to_timedelta(df["dow"], unit="D")

    # -------------------------
    # Monthly
    # -------------------------
    df["month"] = df["date"].dt.to_period("M").dt.start_time

    return df


def save_fact_table():
    parquet_save(create_fact_table(), "fact_table")


def fix_agg_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens MultiIndex / tuple columns into snake_case strings.

    Examples:
    ('unit_sales', 'sum') -> unit_sales_sum
    ('store_nbr', '')     -> store_nbr
    'date'                -> date
    """
    df = df.copy()

    def clean(col: object) -> str:
        # Case 1: tuple column from groupby/agg
        if isinstance(col, tuple):
            parts = [str(p) for p in col if p]
            return "_".join(parts)

        # Case 2: already string
        if isinstance(col, str):
            return col

        # Fallback (should not happen)
        return str(col)

    df.columns = [clean(c) for c in df.columns]
    return df


def save_table(df: pd.DataFrame, name: PreDataset) -> None:
    parquet_save_prepocessed(fix_agg_columns(df), name)


def parquet_save_prepocessed(df: pd.DataFrame, name: PreDataset) -> None:
    df_to_parquet(df, PREPROCESSED_DIR / f"{name.value}.parquet")


def load_fact_table():
    return pd.read_parquet(PREPROCESSED_DIR / "fact_table.parquet")


def load_table(name: PreDataset) -> pd.DataFrame:
    return pd.read_parquet(PREPROCESSED_DIR / f"{name.value}.parquet")


def aggregate(
    df: pd.DataFrame, group_cols: list[str], metrics: list[str]
) -> pd.DataFrame:
    return df.groupby(group_cols).agg(metrics).reset_index()


ALL_METRICS_UNIT_SALES = {
    "unit_sales": ["sum", "mean", "std", "max"],
}


def _make_aggregator(group_keys: list[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Factory: erzeugt eine Aggregationsfunktion für die gegebenen Gruppierungs-Spalten."""

    def aggregator(df: pd.DataFrame) -> pd.DataFrame:
        return aggregate(df, group_keys, ALL_METRICS_UNIT_SALES)

    return aggregator


agg_daily = _make_aggregator(["date"])
agg_weekly = _make_aggregator(["week"])
agg_monthly = _make_aggregator(["month"])
store_daily = _make_aggregator(["store_nbr", "date"])
item_daily = _make_aggregator(["item_nbr", "date"])
store_item_daily = _make_aggregator(["store_nbr", "item_nbr", "date"])
store_weekly = _make_aggregator(["store_nbr", "week"])
item_weekly = _make_aggregator(["item_nbr", "week"])
store_item_weekly = _make_aggregator(
    ["store_nbr", "item_nbr", "year_iso", "week", "week_start"]
)
store_monthly = _make_aggregator(["store_nbr", "month"])
item_monthly = _make_aggregator(["item_nbr", "month"])
store_item_monthly = _make_aggregator(["store_nbr", "item_nbr", "month"])


def save_dailys():
    df_fact = load_fact_table()
    save_table(item_daily(df_fact), PreDataset.ITEM_DAILY)
    save_table(store_daily(df_fact), PreDataset.STORE_DAILY)
    save_table(store_item_daily(df_fact), PreDataset.STORE_ITEM_DAILY)


def save_weeklys():
    df_fact = load_fact_table()
    save_table(item_weekly(df_fact), PreDataset.ITEM_WEEKLY)
    save_table(store_weekly(df_fact), PreDataset.STORE_WEEKLY)
    save_table(store_item_weekly(df_fact), PreDataset.STORE_ITEM_WEEKLY)


def save_monthlys():
    df_fact = load_fact_table()
    save_table(item_monthly(df_fact), PreDataset.ITEM_MONTHLY)
    save_table(store_monthly(df_fact), PreDataset.STORE_MONTHLY)
    save_table(store_item_monthly(df_fact), PreDataset.STORE_ITEM_MONTHLY)


# save_fact_table()
# save_dailys()
# save_weeklys()
# save_monthlys()


# Item Level - Daily, Weekly, Monthly Aggregations
# Store Level - Daily, Weekly, Monthly Aggregations
# Item + Store Level - Daily, Weekly, Monthly Aggregations

# =============================================================================
# Initialisierungs-Logik
# =============================================================================

def create_time_series_tables():
    """
    Erstellt die notwendige Infrastruktur und generiert alle Tabellen.
    """
    for folder in [PREPROCESSED_DIR, METRICS_DIR]:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
    try:
        save_fact_table()
        save_dailys()
        save_weeklys()
        save_monthlys()
    except Exception as e:
        print(f"❌ Fehler bei der Initialisierung: {e}")

# Falls du die Datei als Skript ausführst, wird die Initialisierung gestartet
#create_time_series_tables()
