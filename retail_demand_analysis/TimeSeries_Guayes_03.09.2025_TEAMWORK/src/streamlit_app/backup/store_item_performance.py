import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# 1. PFAD-LOGIK (Chirurgisch hinzugefügt für Unterordner-Struktur)
# Da die Datei in src/streamlit_app/pages/ liegt, gehen wir 3 Ebenen hoch zum Root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table
from streamlit_app.components.charts import render_plotly

# --- PAGE CONFIG ENTFERNT (Wird zentral in app.py gesteuert) ---

# --------------------------------------------------
# Cache heavy computation
# --------------------------------------------------
@st.cache_data(show_spinner=False)
def load_store_item_stats():
    df = load_table(PreDataset.STORE_ITEM_DAILY)

    sold = df.loc[
        df["unit_sales_sum"] > 0,
        ["date", "store_nbr", "item_nbr", "unit_sales_sum"],
    ]

    stats = sold.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        days_sold=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
    )

    return stats


st.header("🐌 Global Store-Item Sales Performance")

stats = load_store_item_stats()

# --------------------------------------------------
# Selection state (Initialisierung)
# --------------------------------------------------
if "selected_store" not in st.session_state:
    st.session_state.selected_store = None
if "selected_item" not in st.session_state:
    st.session_state.selected_item = None

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
with st.sidebar:
    st.subheader("🎯 Highlight & Filter")

    store_pick = st.selectbox(
        "Store Highlight",
        [None, *sorted(stats["store_nbr"].unique())],
        index=0,
    )

    item_pick = st.selectbox(
        "Item Highlight",
        [None, *sorted(stats["item_nbr"].unique())],
        index=0,
    )

    if st.button("Clear selection"):
        st.session_state.selected_store = None
        st.session_state.selected_item = None
        st.rerun()

# Logik-Update des Session States
if store_pick is not None:
    st.session_state.selected_store = store_pick
if item_pick is not None:
    st.session_state.selected_item = item_pick

# --------------------------------------------------
# Highlight mask (Vektorisiert)
# --------------------------------------------------
highlight = np.zeros(len(stats), dtype=bool)

if st.session_state.selected_store is not None:
    highlight |= stats["store_nbr"].values == st.session_state.selected_store

if st.session_state.selected_item is not None:
    highlight |= stats["item_nbr"].values == st.session_state.selected_item

# --------------------------------------------------
# Marker styling (Logik für Farben und Größen)
# --------------------------------------------------
colors = np.where(
    highlight,
    "#58A6FF",  # Fokus-Blau
    "rgba(140, 150, 170, 0.25)",  # Dezentere Hintergrundpunkte
)

sizes = np.where(highlight, 14, 6)
line_widths = np.where(highlight, 1.5, 0)

# --------------------------------------------------
# Plot Erstellung
# --------------------------------------------------
fig = go.Figure(
    go.Scatter(
        x=stats["days_sold"],
        y=stats["total_units"],
        mode="markers",
        marker={
            "size": sizes,
            "color": colors,
            "line": {
                "width": line_widths,
                "color": "white",
            },
        },
        customdata=np.stack(
            [stats["store_nbr"], stats["item_nbr"]],
            axis=1,
        ),
        hovertemplate=(
            "<b>Store %{customdata[0]}</b><br>"
            "Item %{customdata[1]}<br>"
            "Days sold: %{x}<br>"
            "Total units: %{y:,}<extra></extra>"
        ),
    )
)

fig.update_xaxes(type="log", title="Days Sold (log)")
fig.update_yaxes(type="log", title="Total Units Sold (log)")
fig.update_layout(
    height=600,
    dragmode="pan",
    margin=dict(l=0, r=0, t=40, b=0)
)

# Nutzt deine zentrale Plotly-Komponente
render_plotly(fig)

# --------------------------------------------------
# Metrics (UI Logik)
# --------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total pairs", f"{len(stats):,}")
c2.metric("Highlighted", int(highlight.sum()))
c3.metric("Ø Units/Pair", f"{stats['total_units'].mean():.0f}")