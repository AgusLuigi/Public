import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

st.title("🔎 Forecast Candidates (Store-Item) — Table View")

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

    # Aggregation der Metriken
    summary = df.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        n_days_observed=("date", "nunique"),
        total_units=("unit_sales_sum", "sum"),
        n_days_sold=("unit_sales_sum", lambda s: (s > 0).sum()),
        n_days_zero=("unit_sales_sum", lambda s: (s <= 0).sum()),
    )

    summary["sell_through"] = summary["n_days_sold"] / summary["n_days_observed"].clip(lower=1)
    summary["zero_share"] = summary["n_days_zero"] / summary["n_days_observed"].clip(lower=1)
    summary["coverage_global"] = summary["n_days_observed"] / float(global_days)

    return summary

# Daten laden
df_raw = load_store_item_daily()
summary = build_store_item_summary(df_raw)

# LOGIK: FILTER & INTERAKTION
st.sidebar.header("🎯 Filter-Logik")

with st.sidebar:
    min_units = st.number_input("Min. Total Units", value=100, step=50)
    min_coverage = st.slider("Min. Global Coverage", 0.0, 1.0, 0.5)
    
    st.divider()
    sort_col = st.selectbox("Sortieren nach", options=summary.columns, index=2) 
    sort_asc = st.checkbox("Aufsteigend sortieren", value=False)

f_summary = summary[
    (summary["total_units"] >= min_units) & 
    (summary["coverage_global"] >= min_coverage)
].sort_values(sort_col, ascending=sort_asc)
c1, c2, c3 = st.columns(3)
c1.metric("Kandidaten (Gefiltert)", len(f_summary))
c2.metric("Ø Units/Kandidat", f"{f_summary['total_units'].mean():.1f}")
c3.metric("Ø Coverage", f"{f_summary['coverage_global'].mean():.1%}")

st.subheader("Store-Item Summary")
st.dataframe(f_summary,use_container_width=True,hide_index=True)