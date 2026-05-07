import sys
from pathlib import Path
import plotly.express as px
import streamlit as st

# 1. PFAD-LOGIK (Chirurgisch hinzugefügt)
# Wir gehen 3 Ebenen hoch: pages -> streamlit_app -> src -> ROOT
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.append(str(ROOT / "src"))

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.preprocess_data import load_table
from streamlit_app.components.charts import render_plotly

st.title("🏬 Store Performance Analysis")

@st.cache_data(show_spinner="Lade Zeitreihen...")
def load_all_store_data():
    daily = load_table(PreDataset.STORE_DAILY).copy()
    weekly = load_table(PreDataset.STORE_WEEKLY).copy()
    monthly = load_table(PreDataset.STORE_MONTHLY).copy()
    
    # Zeitstempel-Konvertierung direkt beim Laden
    weekly["week_ts"] = weekly["week"].dt.start_time
    monthly["month_ts"] = monthly["month"].dt.to_timestamp()
    
    return daily, weekly, monthly

df_daily, df_weekly, df_monthly = load_all_store_data()

store_id = st.selectbox(
    "Select store",
    sorted(df_daily["store_nbr"].unique()),
    index=0
)

    
df_d_choice = df_daily[df_daily["store_nbr"] == store_id]
df_w_choice = df_weekly[df_weekly["store_nbr"] == store_id]
df_m_choice = df_monthly[df_monthly["store_nbr"] == store_id]

tab1, tab2, tab3 = st.tabs(["Daily", "Weekly", "Monthly"])

with tab1:
    fig_daily = px.line(
        df_d_choice,
        x="date",
        y="unit_sales_sum",
        title=f"Store {store_id} - Daily Sales",
        color_discrete_sequence=["#58A6FF"]
    )
    render_plotly(fig_daily)

with tab2:
    fig_weekly = px.line(
        df_w_choice,
        x="week_ts",
        y="unit_sales_sum",
        title=f"Store {store_id} - Weekly Sales",
        color_discrete_sequence=["#7EE787"]
    )
    render_plotly(fig_weekly)

with tab3:
    fig_monthly = px.line(
        df_m_choice,
        x="month_ts",
        y="unit_sales_sum",
        title=f"Store {store_id} - Monthly Sales",
        color_discrete_sequence=["#BC8CFF"]
    )
    render_plotly(fig_monthly)