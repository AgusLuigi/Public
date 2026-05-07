"""
forecastability_store_item.py

Streamlit-App for Store-Item Forecastability.
Modularized version using central UI components.
"""

from __future__ import annotations

from typing import Literal
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

from Favorita_TSA.utils.forecastability import DAILY_METRICS_PATH, WEEKLY_METRICS_PATH
from Favorita_TSA.utils.config import cfg

# UI COMPONENTS
from streamlit_app.components.filters import render_pattern_filter
from streamlit_app.components.metrics_row import render_metrics_row
from streamlit_app.components.charts import render_plotly

PATTERN_COLOR = {
    "Smooth": "#2ecc71",
    "Erratic": "#f1c40f",
    "Intermittent": "#e67e22",
    "Lumpy": "#e74c3c",
    "Unknown": "#95a5a6",
}

DISPLAY_COLS_DAILY = [
    "store_nbr", 
    "item_label", 
    "emoji", 
    "family", 
    "class", 
    "pattern_label", 
    "perishable", 
    "total_units", 
    "active_span", 
    "periods_sold", 
    "sales_density", 
    "adi", 
    "cv2",
]

DISPLAY_COLS_WEEKLY = [
    "store_nbr", 
    "item_label", 
    "emoji", 
    "family", 
    "class", 
    "daily_pattern", 
    "pattern_label", 
    "perishable", 
    "total_units", 
    "active_span_weeks", 
    "weeks_sold", 
    "sales_density", 
    "adi", 
    "cv2",
]

@st.cache_data(show_spinner=False)
def load_metrics(granularity: str) -> pd.DataFrame:
    path = DAILY_METRICS_PATH if granularity == "daily" else WEEKLY_METRICS_PATH
    try:
        df = pd.read_parquet(path)
        if "perishable" in df.columns:
            df["perishable"] = df["perishable"].astype(bool)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters.get("family"):
        out = out[out["family"].isin(filters["family"])]
    if filters.get("pattern"):
        out = out[out["pattern"].isin(filters["pattern"])]
    if filters.get("perishable") != "All":
        out = out[out["perishable"] == filters["perishable"]]
    if "sales_density" in out.columns:
        out = out[out["sales_density"] >= filters.get("min_density", 0.0)]
    return out

def render_metric_explanation(level: str):
    """Erhält die Erklärungs-Logik für ADI/CV2."""
    span_label = "Tage" if level == "daily" else "Wochen"
    span_col = "active_span" if level == "daily" else "active_span_weeks"
    
    with st.expander("📘 Metrik-Erklärungen"):
        st.markdown(f"""
        **{span_col}**: Lebensspanne ({span_label}) | **sales_density**: Verkaufsfrequenz  
        **adi**: Durchschnittlicher Intervall (höher = seltener) | **cv2**: Volatilität (höher = instabiler)
        ---
        🟢 **Smooth**: Stabil & Häufig | 🟡 **Erratic**: Häufig & Volatil  
        🟠 **Intermittent**: Stabil & Selten | 🔴 **Lumpy**: Volatil & Selten
        """)

def render_summary_metrics(df: pd.DataFrame):
    """Ersetzt manuelle Spalten durch render_metrics_row."""
    render_metrics_row(
        labels=["Angezeigte Zeilen", "Unique Stores", "Unique Items"],
        values=[f"{len(df):,}", str(df["store_nbr"].nunique()), str(df["item_nbr"].nunique())]
    )

def render_distribution_row(df: pd.DataFrame):
    """Zeigt die Pattern-Verteilung kompakt an."""
    patterns = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    labels, values, colors = [], [], []
    
    for p in patterns:
        share = (df["pattern"] == p).mean() if not df.empty else 0
        labels.append(f"{p}")
        values.append(f"{share:.1%}")
        colors.append("normal")
        
    st.write("**Nachfrage-Profile Verteilung:**")
    render_metrics_row(labels=labels, values=values)

def render_tab_content(granularity: Literal["daily", "weekly"]):
    df = load_metrics(granularity)
    if df.empty:
        st.error("Keine Daten vorhanden. Pipeline ausführen!")
        return

    filters = render_pattern_filter(df, key_prefix=granularity)
    df_filtered = apply_filters(df, filters)

    display_cols = DISPLAY_COLS_DAILY if granularity == "daily" else DISPLAY_COLS_WEEKLY
    cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.data_editor(
        df_filtered[cols].sort_values("total_units", ascending=False),
        use_container_width=True,
        hide_index=True,
        disabled=True
    )

    render_summary_metrics(df_filtered)
    st.divider()
    render_distribution_row(df_filtered)
    st.divider()
    render_metric_explanation(granularity)

def main():
    st.title("🧾 Forecastability Dashboard")
    tab_d, tab_w = st.tabs(["📅 Daily", "📆 Weekly"])
    
    with tab_d:
        render_tab_content("daily")
    with tab_w:
        render_tab_content("weekly")

if __name__ == "__main__":
    main()