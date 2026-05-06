"""
app_refactored.py

Streamlit-App für Store-Item Forecastability.
Liest fertige Metriken direkt aus Parquet - keine Berechnungen hier.

Voraussetzung: run_pipelines.py wurde einmalig ausgeführt.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from Favorita_TSA.utils.forecastability import DAILY_METRICS_PATH, WEEKLY_METRICS_PATH
from streamlit_app.components.filters import render_pattern_filter

# =============================================================================
# Seiten-Konfiguration
# =============================================================================

st.set_page_config(layout="wide", page_title="Forecastability Dashboard")


# =============================================================================
# Konstanten (nur UI-relevant)
# =============================================================================

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
    "daily_pattern",  # Pattern aus daily als Kontext
    "pattern_label",  # Pattern auf weekly-Ebene
    "perishable",
    "total_units",
    "active_span_weeks",
    "weeks_sold",
    "sales_density",
    "adi",
    "cv2",
]


# =============================================================================
# Daten laden (gecacht)
# =============================================================================


@st.cache_data(show_spinner=False)
def load_daily_metrics() -> pd.DataFrame:
    df = pd.read_parquet(DAILY_METRICS_PATH)
    if "perishable" in df.columns:
        df["perishable"] = df["perishable"].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_weekly_metrics() -> pd.DataFrame:
    df = pd.read_parquet(WEEKLY_METRICS_PATH)
    if "perishable" in df.columns:
        df["perishable"] = df["perishable"].astype(bool)
    return df


# =============================================================================
# UI-Komponenten
# =============================================================================


def render_filters(df: pd.DataFrame, key_prefix: str) -> dict:
    """Delegiert an die zentrale Filter-Komponente."""
    return render_pattern_filter(df, key_prefix)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Wendet alle Filter auf den DataFrame an."""
    out = df.copy()

    if filters["family"]:
        out = out[out["family"].isin(filters["family"])]

    if filters["pattern"]:
        out = out[out["pattern"].isin(filters["pattern"])]

    if filters["perishable"] != "All":
        out = out[out["perishable"] == filters["perishable"]]

    out = out[out["sales_density"] >= filters["min_density"]]

    return out


def render_metric_guide(level: str) -> None:
    """Expandierbarer Metrik-Guide, angepasst an daily vs. weekly."""
    span_label = "Tage" if level == "daily" else "Wochen"
    sold_label = "days_sold" if level == "daily" else "weeks_sold"
    span_col = "active_span" if level == "daily" else "active_span_weeks"

    with st.expander("📘 Metrik-Erklärungen"):
        st.markdown(
            f"""
**{span_col}**
Anzahl {span_label} zwischen erstem und letztem Verkauf → Lebensspanne des Artikels im Store

**{sold_label}**
Anzahl {span_label} mit tatsächlichen Verkäufen

**sales_density**
{sold_label} / {span_col} → je höher, desto regelmäßiger

**zero_rate**
Anteil {span_label} ohne Verkäufe → je höher, desto intermittierender

**adi (Average Demand Interval)**
Durchschnittlicher Abstand zwischen Verkäufen → höher = seltener

**cv2 (Demand Volatility)**
Quadrierter Variationskoeffizient → höher = unstabiler

**total_units**
Gesamtvolumen über den gesamten Zeitraum

---
🟢 Smooth → niedriger ADI + niedriges CV² → gut vorhersagbar  
🟡 Erratic → häufig aber volatil  
🟠 Intermittent → selten aber stabil  
🔴 Lumpy → selten und volatil → schwer vorhersagbar
        """
        )


def render_summary_row(df: pd.DataFrame) -> None:
    """Kennzahlen-Zeile unter der Tabelle."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Angezeigte Zeilen", f"{len(df):,}")
    c2.metric("Unique Stores", df["store_nbr"].nunique())
    c3.metric("Unique Items", df["item_nbr"].nunique())


def render_pattern_distribution(df: pd.DataFrame) -> None:
    """Farbige Pattern-Anteile in vier Spalten."""
    patterns = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    cols = st.columns(4)

    for col, pattern in zip(cols, patterns, strict=False):
        share = (df["pattern"] == pattern).mean()
        color = PATTERN_COLOR[pattern]
        emoji = (
            df.loc[df["pattern"] == pattern, "pattern_emoji"].iloc[0]
            if (df["pattern"] == pattern).any()
            else "⚪"
        )
        with col:
            st.markdown(
                f"<span style='color:{color}; font-weight:700;'>{emoji} {pattern}</span>",
                unsafe_allow_html=True,
            )
            st.metric(label="", value=f"{share:.1%}")


# =============================================================================
# Tab-Render-Funktionen
# =============================================================================


def render_aggregation_tab(granularity: Literal["daily", "weekly"]) -> None:
    """Rendert den Forecastability-Tab für daily oder weekly Granularität."""
    is_daily = granularity == "daily"
    icon = "📅" if is_daily else "📆"
    label = "Daily" if is_daily else "Weekly"
    loader = load_daily_metrics if is_daily else load_weekly_metrics
    display_cols = DISPLAY_COLS_DAILY if is_daily else DISPLAY_COLS_WEEKLY

    st.subheader(f"{icon} Store · Item · {label} - Forecastability")

    try:
        df = loader()
    except FileNotFoundError:
        st.error(
            "⚠️ Keine Daten gefunden. Bitte zuerst `python run_pipelines.py` ausführen."
        )
        return

    filters = render_filters(df, key_prefix=granularity)
    df_view = apply_filters(df, filters)
    cols = [c for c in display_cols if c in df_view.columns]

    st.data_editor(
        df_view[cols].sort_values("total_units", ascending=False),
        use_container_width=True,
        hide_index=True,
        disabled=cols,
    )

    render_summary_row(df_view)
    st.divider()
    render_pattern_distribution(df_view)


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.title("🧾 Forecastability Dashboard")

    tab_daily, tab_weekly = st.tabs(["📅 Daily", "📆 Weekly"])

    with tab_daily:
        render_aggregation_tab("daily")

    with tab_weekly:
        render_aggregation_tab("weekly")


if __name__ == "__main__":
    main()
