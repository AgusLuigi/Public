import numpy as np
import plotly.graph_objects as go
import streamlit as st

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table


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
# Selection state
# --------------------------------------------------
if "selected_store" not in st.session_state:
    st.session_state.selected_store = None

if "selected_item" not in st.session_state:
    st.session_state.selected_item = None


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
with st.sidebar:
    st.subheader("🎯 Highlight")

    store_pick = st.selectbox(
        "Store",
        [None, *sorted(stats["store_nbr"].unique())],
        index=0,
    )

    item_pick = st.selectbox(
        "Item",
        [None, *sorted(stats["item_nbr"].unique())],
        index=0,
    )

    if st.button("Clear selection"):
        st.session_state.selected_store = None
        st.session_state.selected_item = None

# Update session state
if store_pick is not None:
    st.session_state.selected_store = store_pick
    st.session_state.selected_item = None

if item_pick is not None:
    st.session_state.selected_item = item_pick
    st.session_state.selected_store = None


# --------------------------------------------------
# Highlight mask (vectorized, fast)
# --------------------------------------------------
highlight = np.zeros(len(stats), dtype=bool)

if st.session_state.selected_store is not None:
    highlight |= stats["store_nbr"].values == st.session_state.selected_store

if st.session_state.selected_item is not None:
    highlight |= stats["item_nbr"].values == st.session_state.selected_item


# --------------------------------------------------
# Marker styling (KEY PART)
# --------------------------------------------------
colors = np.where(
    highlight,
    "#58A6FF",  # focus blue
    "rgba(140, 150, 170, 0.35)",  # visible gray-blue
)

sizes = np.where(
    highlight,
    14,
    6,
)

line_widths = np.where(
    highlight,
    2,
    0,
)

opacities = np.where(
    highlight,
    1.0,
    0.6,
)

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig = go.Figure(
    go.Scatter(
        x=stats["days_sold"],
        y=stats["total_units"],
        mode="markers",
        marker={
            "size": sizes,
            "color": colors,
            "opacity": opacities,
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
            "Store %{customdata[0]}<br>"
            "Item %{customdata[1]}<br>"
            "Days sold %{x}<br>"
            "Total units %{y}<extra></extra>"
        ),
    )
)

fig.update_xaxes(type="log", title="Days Sold (log)")
fig.update_yaxes(type="log", title="Total Units Sold (log)")
fig.update_layout(
    title="Global Store-Item Sales Performance",
    dragmode="pan",
)

# --------------------------------------------------
# Render
# --------------------------------------------------
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
c1, c2 = st.columns(2)
c1.metric("Total store-item pairs", f"{len(stats):,}")
c2.metric(
    "Highlighted",
    int(highlight.sum()),
)
