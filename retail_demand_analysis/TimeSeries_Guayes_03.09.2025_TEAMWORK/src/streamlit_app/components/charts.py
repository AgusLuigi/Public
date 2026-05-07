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
    """
    Rendert eine Plotly-Figure mit konsistenten Einstellungen.
    Zentralisiert das Styling, damit alle Charts in der App gleich aussehen.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
) -> None:
    """Rendert ein Liniendiagramm via Plotly Express."""
    fig = px.line(df, x=x, y=y, color=color, title=title)
    render_plotly(fig)

# Helpful for Store-Item analyses
def render_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    title: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """Rendert einen Scatterplot, ideal für Performance-Checks."""
    fig = px.scatter(df, x=x, y=y, color=color, title=title, log_x=log_x, log_y=log_y)
    render_plotly(fig)