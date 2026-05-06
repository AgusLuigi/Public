import pandas as pd
import streamlit as st

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Forecast Candidates", layout="wide")
st.title("🔎 Forecast Candidates (Store-Item) — Table View")


# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
@st.cache_data
def load_and_prepare():
    df = load_table(PreDataset.STORE_ITEM_DAILY).copy()

    return df


@st.cache_data(show_spinner=True)
def load_store_item_daily() -> pd.DataFrame:
    df = load_table(PreDataset.STORE_ITEM_DAILY).copy()

    # Ensure types
    df["date"] = pd.to_datetime(df["date"])
    # Guard against missing columns
    required = {"store_nbr", "item_nbr", "date", "unit_sales_sum"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in STORE_ITEM_DAILY: {sorted(missing)}")

    return df


@st.cache_data(show_spinner=True)
def build_store_item_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Step 1: global historical depth (how many days exist in the dataset)
    global_start = df["date"].min()
    global_end = df["date"].max()
    global_days = (global_end - global_start).days + 1  # inclusive

    pair_days = df.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        n_days_observed=("date", "nunique")
    )

    pair_sales = df.groupby(["store_nbr", "item_nbr"], as_index=False).agg(
        n_days_sold=("unit_sales_sum", lambda s: (s > 0).sum()),
        n_days_zero=("unit_sales_sum", lambda s: (s <= 0).sum()),
    )

    summary = pair_days.merge(pair_sales, on=["store_nbr", "item_nbr"], how="left")

    summary["sell_through"] = summary["n_days_sold"] / summary["n_days_observed"].clip(
        lower=1
    )
    summary["zero_share"] = summary["n_days_zero"] / summary["n_days_observed"].clip(
        lower=1
    )
    summary["coverage_global"] = summary["n_days_observed"] / float(global_days)

    return summary


df_store_item_daily = load_store_item_daily()

summary = build_store_item_summary(df_store_item_daily)

st.subheader("Store-Item Summary")
st.dataframe(
    summary,
    use_container_width=True,
)
