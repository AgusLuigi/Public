"""
date_utils.py

Zentrale Datum/Zeit-Hilfsfunktionen für das gesamte Projekt.
Konsolidiert verstreute Period-zu-Timestamp-Konvertierungen aus
store_item_behavior.py, multi_stores.py und preprocess_data.py.
"""

from __future__ import annotations

import pandas as pd


def period_to_timestamp(series: pd.Series) -> pd.Series:
    """Konvertiert eine Period- oder String-Series zu normalisierten Timestamps."""
    if pd.api.types.is_period_dtype(series):
        return series.dt.start_time
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def normalize_time_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Vereinheitlicht eine Zeit-Spalte im DataFrame zu datetime64[ns].
    Unterstützt Period, String und datetime Eingaben.
    Zeilen mit ungültigem Datum werden entfernt.
    """
    out = df.copy()
    if col not in out.columns:
        return out
    out[col] = period_to_timestamp(out[col])
    return out.dropna(subset=[col])


def get_date_col(df: pd.DataFrame) -> str:
    """
    Gibt den Namen der vorhandenen Zeit-Spalte zurück.
    Prüft in Reihenfolge: 'date', 'week_start', 'week', 'month'.
    Wirft ValueError wenn keine gefunden wird.
    """
    for col in ("date", "week_start", "week", "month"):
        if col in df.columns:
            return col
    raise ValueError(
        f"Keine bekannte Zeit-Spalte gefunden. Vorhandene Spalten: {list(df.columns)}"
    )
