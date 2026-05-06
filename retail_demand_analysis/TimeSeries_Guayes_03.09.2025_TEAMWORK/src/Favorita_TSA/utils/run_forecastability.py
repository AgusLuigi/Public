"""
run_pipelines.py

Fuehrt beide Pipelines aus (daily + weekly) und speichert die
Metriken als Parquet unter data/metrics/.

Reihenfolge ist bewusst: Daily muss zuerst laufen, da die Weekly-Pipeline
das daily-Ergebnis als Filter verwendet (nur Intermittent + Lumpy).

Aufruf:
    python run_pipelines.py
"""

from Favorita_TSA.utils.data_loader import parquet_loader
from Favorita_TSA.utils.dataset import Dataset, PreDataset
from Favorita_TSA.utils.forecastability import run_daily_pipeline, run_weekly_pipeline
from Favorita_TSA.utils.preprocess_data import load_table

FAMILY_EMOJI = {
    "GROCERY": "🥫",
    "BEVERAGES": "🥤",
    "CLEANING": "🧽",
    "DAIRY": "🥛",
    "PRODUCE": "🥕",
    "MEATS": "🥩",
    "BREAD/BAKERY": "🍞",
    "DELI": "🧀",
    "PERSONAL CARE": "🧴",
    "HOME CARE": "🏠",
}


def load_item_metadata():
    items = parquet_loader(Dataset.ITEMS)
    items["emoji"] = items["family"].map(FAMILY_EMOJI).fillna("📦")
    items["item_label"] = items["item_nbr"].astype(str)
    return items[["item_nbr", "item_label", "family", "class", "emoji", "perishable"]]


if __name__ == "__main__":
    print("Lade Rohdaten ...")
    df_daily = load_table(PreDataset.STORE_ITEM_DAILY)
    df_weekly = load_table(PreDataset.STORE_ITEM_WEEKLY)
    item_meta = load_item_metadata()

    # Daily muss zuerst laufen - Ergebnis wird an Weekly weitergegeben
    print("\nDaily Pipeline ...")
    daily_metrics = run_daily_pipeline(df_daily, item_metadata=item_meta)

    # Weekly nur fuer Intermittent + Lumpy aus daily
    print("\nWeekly Pipeline (nur Intermittent + Lumpy aus daily) ...")
    run_weekly_pipeline(
        df_weekly,
        daily_metrics=daily_metrics,
        item_metadata=item_meta,
    )

    print("\nFertig - beide Parquet-Dateien sind aktuell.")
