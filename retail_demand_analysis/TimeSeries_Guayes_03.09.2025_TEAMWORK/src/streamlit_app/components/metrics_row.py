"""
metrics_row.py

Reusable metric row component.
"""

from __future__ import annotations
import streamlit as st

def render_metrics_row(
    labels: list[str],
    values: list[str | float | int],
    deltas: list[str | float | int | None] | None = None,
    delta_colors: list[str] | None = None,
    help_texts: list[str | None] | None = None, 
) -> None:
    """
    Rendert eine Zeile aus Metriken nebeneinander.
    """
    if len(labels) == 0:
        return

    cols = st.columns(len(labels))
    
    for i, col in enumerate(cols):
        label = labels[i]
        value = values[i]
        delta = deltas[i] if (deltas and i < len(deltas)) else None
        d_color = delta_colors[i] if (delta_colors and i < len(delta_colors)) else "normal"
        h_text = help_texts[i] if (help_texts and i < len(help_texts)) else None

        col.metric(
            label=label, 
            value=value, 
            delta=delta, 
            delta_color=d_color,
            help=h_text # Hilft dem Nutzer, die Metrik zu verstehen
        )