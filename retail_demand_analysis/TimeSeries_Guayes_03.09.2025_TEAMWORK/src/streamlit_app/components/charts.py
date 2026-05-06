"""
charts.py

Zentrale Chart-Render-Funktionen für Streamlit-Pages.
Ersetzt 12+ verstreute `st.plotly_chart(fig, use_container_width=True)` Aufrufe.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_plotly(fig: go.Figure) -> None:
    """Rendert eine Plotly-Figure mit konsistenten Einstellungen."""
    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str,
    title: str,
) -> None:
    """Rendert ein Liniendiagramm via Plotly Express."""
    fig = px.line(df, x=x, y=y, color=color, title=title)
    render_plotly(fig)
