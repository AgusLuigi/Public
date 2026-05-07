from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# 1. PFAD-LOGIK (Chirurgisch hinzugefügt für Unterordner-Struktur)
# Da diese Datei in src/streamlit_app/pages/ liegt, gehen wir 3 Ebenen hoch zum Root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

# 2. UI SETUP (Config entfernt, da diese zentral in app.py gesteuert wird)
st.title("🔎 Forecast Candidates (Store-Item) — Table View")

# -----------------------------
# Cached data loading
# -----------------------------
@st.cache_data(show_spinner=True)
def load_store_item_daily() -> pd.DataFrame:
    df = load_table(PreDataset.STORE_ITEM_DAILY).copy()
    df["date"] = pd.to_datetime(df["date"])
    
    required = {"store_nbr", "item_nbr", "date", "unit_sales_sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in STORE_ITEM_DAILY: {sorted(missing)}")
    return df

@st.cache_data(show_spinner=True)
def build_store_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    global_start = df["date"].min()
    global_end = df["date"].max()
    global_days = (global_end - global_start).days + 1

    g = df.groupby(["store_nbr", "item_nbr"], as_index=False)

    summary = g.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_days_observed=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
        mean_units=("unit_sales_sum", "mean"),
        std_units=("unit_sales_sum", "std"),
        max_units=("unit_sales_sum", "max"),
        n_days_sold=("unit_sales_sum", lambda s: (s > 0).sum()),
        n_days_zero=("unit_sales_sum", lambda s: (s <= 0).sum()),
    )

    summary["active_span_days"] = (summary["last_date"] - summary["first_date"]).dt.days + 1
    summary["coverage_global"] = summary["n_days_observed"] / float(global_days)
    summary["sell_through"] = summary["n_days_sold"] / summary["n_days_observed"].clip(lower=1)
    summary["zero_share"] = summary["n_days_zero"] / summary["n_days_observed"].clip(lower=1)
    summary["cv"] = summary["std_units"] / summary["mean_units"].replace(0, pd.NA)

    summary["std_units"] = summary["std_units"].fillna(0.0)
    summary = summary.sort_values(["total_units"], ascending=False).reset_index(drop=True)
    summary["first_date"] = summary["first_date"].dt.date
    summary["last_date"] = summary["last_date"].dt.date

    return summary

# Daten laden
df_raw = load_store_item_daily()
summary = build_store_item_summary(df_raw)

# -----------------------------
# Sidebar filters
# -----------------------------
with st.sidebar:
    st.header("🎯 Filter-Logik")
    min_total_units = st.number_input("Min total units", min_value=0.0, value=0.0, step=100.0)
    min_days_sold = st.number_input("Min days sold", min_value=0, value=0, step=10)
    min_coverage = st.slider("Min global coverage", 0.0, 1.0, 0.0, 0.01)
    max_zero_share = st.slider("Max zero share", 0.0, 1.0, 1.0, 0.01)

    store_pick = st.selectbox("Store (optional)", options=[None, *sorted(summary["store_nbr"].unique())], index=0)
    item_pick = st.selectbox("Item (optional)", options=[None, *sorted(summary["item_nbr"].unique())], index=0)

    st.divider()
    top_n = st.slider("Rows to show", 100, 5000, 500, 100)
    sort_by = st.selectbox("Sort by", options=["total_units", "n_days_sold", "sell_through", "coverage_global", "zero_share", "cv", "max_units"], index=0)
    sort_asc = st.checkbox("Sort ascending", value=False)

# -----------------------------
# Apply filters
# -----------------------------
f = summary.copy()

f = f.loc[f["total_units"] >= float(min_total_units)]
f = f.loc[f["n_days_sold"] >= int(min_days_sold)]
f = f.loc[f["coverage_global"] >= float(min_coverage)]
f = f.loc[f["zero_share"] <= float(max_zero_share)]

if store_pick is not None:
    f = f.loc[f["store_nbr"] == store_pick]

if item_pick is not None:
    f = f.loc[f["item_nbr"] == item_pick]

# Sort + limit
f = f.sort_values(sort_by, ascending=sort_asc).head(int(top_n)).reset_index(drop=True)

# -----------------------------
# Headline metrics
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total pairs (all)", f"{len(summary):,}")
c2.metric("After filters", f"{len(f):,}")
c3.metric("Unique stores", f"{f['store_nbr'].nunique():,}")
c4.metric("Unique items", f"{f['item_nbr'].nunique():,}")

st.divider()

# -----------------------------
# Table
# -----------------------------
st.subheader("📋 Store-Item Summary Table")

# Choose columns for display (keep it readable)
display_cols = [
    "store_nbr",
    "item_nbr",
    "first_date",
    "last_date",
    "active_span_days",
    "n_days_observed",
    "n_days_sold",
    "sell_through",
    "zero_share",
    "coverage_global",
    "total_units",
    "mean_units",
    "std_units",
    "cv",
    "max_units",
]

st.dataframe(f[display_cols],use_container_width=True,hide_index=True,)

st.caption(
    "Tip: use filters to find candidates that have (1) enough sold days, "
    "(2) low zero_share, and (3) reasonable coverage. Those are usually forecastable."
)