"""
metrics_row.py

Komponente für eine Zeile aus st.metric()-Feldern.
Ersetzt duplizierte 5-Spalten-Metrik-Blöcke in model_training.py und store_item_behavior.py.
"""

from __future__ import annotations

import streamlit as st


def render_metrics_row(
    labels: list[str],
    values: list[str],
    deltas: list[str | None] | None = None,
    delta_colors: list[str] | None = None,
) -> None:
    """
    Rendert eine Zeile aus Metriken nebeneinander.

    Parameters
    ----------
    labels       : Beschriftungen der Metriken
    values       : Anzuzeigende Werte (bereits formatiert als Strings)
    deltas       : Optionale Delta-Werte (gleiche Länge wie labels)
    delta_colors : Optionale Delta-Farben ('normal', 'inverse', 'off')
    """
    cols = st.columns(len(labels))
    for i, (col, label, value) in enumerate(zip(cols, labels, values, strict=False)):
        delta = deltas[i] if deltas else None
        delta_color = delta_colors[i] if delta_colors else "normal"
        col.metric(label=label, value=value, delta=delta, delta_color=delta_color)
