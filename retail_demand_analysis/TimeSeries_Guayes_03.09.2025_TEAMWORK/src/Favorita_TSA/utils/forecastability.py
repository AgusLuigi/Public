"""
pipeline.py

Forecastability metrics pipeline für zwei Aggregationsebenen:
  - store_item_daily   (date-Spalte)
  - store_item_weekly  (week, week_start, year_iso-Spalten)

Beide Pipelines schreiben ihr Ergebnis als Parquet,
das die Streamlit-App direkt einliest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Pfade
# =============================================================================
from Favorita_TSA.utils.config import cfg
from Favorita_TSA.utils.paths import METRICS_DIR

METRICS_DIR.mkdir(parents=True, exist_ok=True)

DAILY_METRICS_PATH = METRICS_DIR / "store_item_daily_metrics.parquet"
WEEKLY_METRICS_PATH = METRICS_DIR / "store_item_weekly_metrics.parquet"

# Croston-Schwellwerte (aus config.yaml)
ADI_THRESHOLD = cfg.croston.adi_threshold
CV2_THRESHOLD = cfg.croston.cv2_threshold

PATTERN_EMOJI = {
    "Smooth": "🟢",
    "Erratic": "🟡",
    "Intermittent": "🟠",
    "Lumpy": "🔴",
    "Unknown": "⚪",
}


# =============================================================================
# Interne Pipeline-Schritte
# =============================================================================


def _aggregate_base_metrics(
    df: pd.DataFrame,
    groupby_cols: list[str],
    date_col: str,
    sales_col: str,
) -> pd.DataFrame:
    """
    Aggregiert df nach groupby_cols und berechnet Basisstatistiken.

    Konvertiert date_col intern nach datetime64, damit first_sale/last_sale
    immer als Timestamp ankommen - unabhaengig davon ob der Aufrufer
    Strings oder datetime-Objekte uebergibt.

    Gibt zurueck:
        first_sale, last_sale, periods_sold, total_units, mean_sales, std_sales
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    agg = (
        df.groupby(groupby_cols)
        .agg(
            first_sale=(date_col, "min"),
            last_sale=(date_col, "max"),
            periods_sold=(date_col, "count"),
            total_units=(sales_col, "sum"),
            mean_sales=(sales_col, "mean"),
            std_sales=(sales_col, "std"),
        )
        .reset_index()
    )
    return agg


def _calculate_lifecycle_metrics(
    agg: pd.DataFrame,
    span_unit: str,
) -> pd.DataFrame:
    """
    Berechnet active_span, sales_density, zero_rate.

    span_unit: 'days' für daily-Pipeline, 'weeks' für weekly-Pipeline
    """
    agg = agg.copy()

    if span_unit == "days":
        agg["active_span"] = (agg["last_sale"] - agg["first_sale"]).dt.days + 1
    elif span_unit == "weeks":
        agg["active_span"] = (agg["last_sale"] - agg["first_sale"]).dt.days // 7 + 1
    else:
        raise ValueError(
            f"span_unit muss 'days' oder 'weeks' sein, nicht {span_unit!r}"
        )

    agg["sales_density"] = agg["periods_sold"] / agg["active_span"]
    agg["zero_rate"] = 1 - agg["sales_density"]

    return agg


def _calculate_demand_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    """Berechnet ADI und CV²."""
    agg = agg.copy()

    agg["adi"] = agg["active_span"] / agg["periods_sold"]
    agg["cv2"] = (agg["std_sales"] / agg["mean_sales"]) ** 2

    agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    return agg


def _classify_demand_pattern(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Croston-Klassifikation basierend auf ADI und CV².

    Fügt hinzu: pattern, pattern_emoji, pattern_label
    """
    agg = agg.copy()

    agg["pattern"] = np.select(
        [
            (agg["adi"] <= ADI_THRESHOLD) & (agg["cv2"] <= CV2_THRESHOLD),
            (agg["adi"] <= ADI_THRESHOLD) & (agg["cv2"] > CV2_THRESHOLD),
            (agg["adi"] > ADI_THRESHOLD) & (agg["cv2"] <= CV2_THRESHOLD),
            (agg["adi"] > ADI_THRESHOLD) & (agg["cv2"] > CV2_THRESHOLD),
        ],
        ["Smooth", "Erratic", "Intermittent", "Lumpy"],
        default="Unknown",
    )

    agg["pattern_emoji"] = agg["pattern"].map(PATTERN_EMOJI)
    agg["pattern_label"] = agg["pattern_emoji"] + " " + agg["pattern"]

    return agg


def _filter_sparse_patterns(
    df: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    patterns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reduziert df auf store-item-Kombinationen, die in daily_metrics
    einem der angegebenen Muster entsprechen.

    Standardmäßig: Intermittent + Lumpy (die schwer vorhersagbaren Fälle).

    Parameters
    ----------
    df            : DataFrame, der gefiltert werden soll (z.B. df_weekly)
    daily_metrics : Ergebnis von run_daily_pipeline() mit "pattern"-Spalte
    patterns      : Liste der Pattern-Namen; None -> ["Intermittent", "Lumpy"]
    """
    if patterns is None:
        patterns = ["Intermittent", "Lumpy"]

    sparse_keys = daily_metrics.loc[
        daily_metrics["pattern"].isin(patterns),
        ["store_nbr", "item_nbr"],
    ]

    n_before = len(df)
    filtered = df.merge(sparse_keys, on=["store_nbr", "item_nbr"], how="inner")
    n_after = len(filtered)

    print(
        f"Filter auf {patterns}: "
        f"{n_before:,} -> {n_after:,} store-item-Kombinationen "
        f"({n_after / n_before:.1%})"
    )
    return filtered


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Speichert df als Parquet."""
    df.to_parquet(path, index=False)
    print(f"Gespeichert: {path}  ({df.shape[0]:,} Zeilen x {df.shape[1]} Spalten)")


# =============================================================================
# Öffentliche Pipeline-Funktionen
# =============================================================================


def run_daily_pipeline(
    df_daily: pd.DataFrame,
    item_metadata: pd.DataFrame | None = None,
    sales_col: str = "unit_sales_sum",
    output_path: Path = DAILY_METRICS_PATH,
) -> pd.DataFrame:
    """
    Berechnet Forecastability-Metriken auf store_item_daily-Ebene
    und speichert das Ergebnis als Parquet.

    Erwartet df_daily mit Spalten:
        store_nbr, item_nbr, date, <sales_col>

    Optionale Anreicherung via item_metadata (muss 'item_nbr' enthalten).

    Returns
    -------
    pd.DataFrame mit Spalten:
        store_nbr, item_nbr,
        first_sale, last_sale, active_span, periods_sold,
        total_units, mean_sales, std_sales,
        sales_density, zero_rate, adi, cv2,
        pattern, pattern_emoji, pattern_label,
        + alle Spalten aus item_metadata
    """
    groupby_cols = ["store_nbr", "item_nbr"]
    date_col = "date"

    df = df_daily.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    metrics = _aggregate_base_metrics(df, groupby_cols, date_col, sales_col)
    metrics = _calculate_lifecycle_metrics(metrics, span_unit="days")
    metrics = _calculate_demand_metrics(metrics)
    metrics = _classify_demand_pattern(metrics)

    if item_metadata is not None:
        metrics = _enrich_item_metadata(metrics, item_metadata)

    _save_parquet(metrics, output_path)
    return metrics


def run_weekly_pipeline(
    df_weekly: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    item_metadata: pd.DataFrame | None = None,
    sales_col: str = "unit_sales_sum",
    sparse_patterns: list[str] | None = None,
    output_path: Path = WEEKLY_METRICS_PATH,
) -> pd.DataFrame:
    """
    Berechnet Forecastability-Metriken auf store_item_weekly-Ebene
    und speichert das Ergebnis als Parquet.

    Beschraenkt sich auf store-item-Kombinationen, die in der daily-Pipeline
    als schwer vorhersagbar klassifiziert wurden (Standard: Intermittent + Lumpy).

    Parameters
    ----------
    df_weekly       : Aggregiertes Weekly-DataFrame mit Spalten:
                      store_nbr, item_nbr, week, week_start, year_iso, <sales_col>
    daily_metrics   : Ergebnis von run_daily_pipeline() — liefert den Pattern-Filter
    item_metadata   : Optionale Item-Metadaten (muss "item_nbr" enthalten)
    sales_col       : Name der Verkaufsspalte
    sparse_patterns : Pattern-Namen die behalten werden;
                      None -> ["Intermittent", "Lumpy"]
    output_path     : Ziel-Parquet-Pfad

    Returns
    -------
    pd.DataFrame mit Spalten:
        store_nbr, item_nbr,
        first_week_start, last_week_start, active_span_weeks, weeks_sold,
        total_units, mean_sales, std_sales,
        sales_density, zero_rate, adi, cv2,
        pattern, pattern_emoji, pattern_label,
        daily_pattern,           <- Pattern aus der daily-Pipeline (zur Referenz)
        + alle Spalten aus item_metadata
    """
    groupby_cols = ["store_nbr", "item_nbr"]
    date_col = "week_start"

    # Schritt 1: Nur schwer vorhersagbare Kombinationen aus daily übernehmen
    df = _filter_sparse_patterns(df_weekly, daily_metrics, patterns=sparse_patterns)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    # Schritt 2-5: Standard-Pipeline
    metrics = _aggregate_base_metrics(df, groupby_cols, date_col, sales_col)
    metrics = _calculate_lifecycle_metrics(metrics, span_unit="weeks")
    metrics = _calculate_demand_metrics(metrics)
    metrics = _classify_demand_pattern(metrics)

    # Sprechende Spaltennamen für wöchentliche Ebene
    metrics = metrics.rename(
        columns={
            "active_span": "active_span_weeks",
            "periods_sold": "weeks_sold",
            "first_sale": "first_week_start",
            "last_sale": "last_week_start",
        }
    )

    # Schritt 6: Daily-Pattern als Referenzspalte hinzufügen
    daily_ref = daily_metrics[["store_nbr", "item_nbr", "pattern"]].rename(
        columns={"pattern": "daily_pattern"}
    )
    metrics = metrics.merge(daily_ref, on=["store_nbr", "item_nbr"], how="left")

    if item_metadata is not None:
        metrics = _enrich_item_metadata(metrics, item_metadata)

    _save_parquet(metrics, output_path)
    return metrics


# =============================================================================
# Hilfs-Funktion: Item-Metadaten anreichern
# =============================================================================


def _enrich_item_metadata(
    metrics: pd.DataFrame,
    item_metadata: pd.DataFrame,
    item_key: str = "item_nbr",
) -> pd.DataFrame:
    """
    Left-Join von item_metadata auf metrics über item_key.
    Spalten, die bereits in metrics vorhanden sind, werden übersprungen.
    """
    extra_cols = [
        c for c in item_metadata.columns if c != item_key and c not in metrics.columns
    ]
    if not extra_cols:
        return metrics

    return metrics.merge(
        item_metadata[[item_key, *extra_cols]],
        on=item_key,
        how="left",
    )
