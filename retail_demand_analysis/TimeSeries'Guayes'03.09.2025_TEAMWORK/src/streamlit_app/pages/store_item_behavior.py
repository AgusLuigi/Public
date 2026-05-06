import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from Favorita_TSA.utils.dataset import PreDataset
from Favorita_TSA.utils.date_utils import normalize_time_col as _normalize_time_col
from Favorita_TSA.utils.paths import PREPROCESSED_DIR
from Favorita_TSA.utils.preprocess_data import load_table
from Favorita_TSA.viz.color_manager import ColorManager
from streamlit_app.components.charts import render_plotly

color_manager = ColorManager()

st.set_page_config(layout="wide")


# =====================================================
# Load data
# =====================================================
df_daily = load_table(PreDataset.STORE_ITEM_DAILY)
df_weekly = load_table(PreDataset.STORE_ITEM_WEEKLY)


# holiday-enriched parquets
def _load_preprocessed(filename: str) -> pd.DataFrame:
    p = PREPROCESSED_DIR / filename
    if not p.exists():
        raise FileNotFoundError(f"Missing parquet: {p}")
    return pd.read_parquet(p)


df_daily_hol = _load_preprocessed("store_item_daily_holiday.parquet")
df_weekly_hol = _load_preprocessed("store_item_weekly_holiday.parquet")


# =====================================================
# Helpers
# =====================================================


def to_dense_index(ts: pd.Series, freq: str) -> pd.Series:
    if ts.empty:
        return ts
    ts = ts.sort_index()
    full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
    return ts.reindex(full_idx, fill_value=0.0)


def get_series_dense(
    df: pd.DataFrame, store: int, item: int, date_col: str, value_col: str, freq: str
) -> pd.Series:
    d = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
    if d.empty:
        return pd.Series(dtype=float)

    d = _normalize_time_col(d, date_col)
    ts = d.groupby(date_col)[value_col].sum().sort_index().astype(float)
    return to_dense_index(ts, freq=freq)


def get_holiday_flags_dense(
    df_hol: pd.DataFrame,
    store: int,
    item: int,
    date_col: str,
    freq: str,
    flag_cols: tuple[str, str, str] = (
        "is_holiday_or_event",
        "pre_holiday",
        "post_holiday",
    ),
) -> pd.DataFrame:
    d = df_hol[(df_hol["store_nbr"] == store) & (df_hol["item_nbr"] == item)].copy()
    if d.empty:
        return pd.DataFrame(columns=list(flag_cols))

    d = _normalize_time_col(d, date_col)

    for c in flag_cols:
        if c not in d.columns:
            d[c] = False

    g = d.groupby(date_col)[list(flag_cols)].any().sort_index()

    full_idx = pd.date_range(start=g.index.min(), end=g.index.max(), freq=freq)
    g = g.reindex(full_idx, fill_value=False)

    for c in flag_cols:
        g[c] = g[c].astype(bool)

    return g


def seasonality_strength(series: pd.Series, lag: int) -> float:
    if series.empty or len(series) < lag * 2:
        return np.nan
    vals = acf(series.values, nlags=lag, fft=True)
    return float(vals[lag])


def stl_decompose(series: pd.Series, period: int):
    stl = STL(series, period=period, robust=True)
    res = stl.fit()
    return res.trend, res.seasonal, res.resid


def build_holiday_badges(hol_flags: pd.DataFrame) -> np.ndarray:
    """
    Returns customdata array with HTML badges:
      [pre_badge, holiday_badge, post_badge]
    """

    def badge(arr: np.ndarray) -> np.ndarray:
        green = (
            "background:rgba(120,220,150,0.25);"
            "border:1px solid rgba(120,220,150,0.45);"
            f"color:{color_manager.get_colors().green_brand}"
        )
        red = (
            "background:rgba(255,120,120,0.20);"
            "border:1px solid rgba(255,120,120,0.45);"
            f"color:{color_manager.get_colors().red_brand};"
        )

        styles = np.where(arr, green, red)
        labels = np.where(arr, "TRUE", "FALSE")

        return np.array(
            [
                f"<span style='display:inline-block; min-width:64px; "
                f"text-align:center; padding:2px 8px; border-radius:999px; {s}'>"
                f"{t}</span>"
                for s, t in zip(styles, labels, strict=False)
            ],
            dtype=object,
        )

    pre = badge(hol_flags["pre_holiday"].to_numpy(dtype=bool))
    hol = badge(hol_flags["is_holiday_or_event"].to_numpy(dtype=bool))
    post = badge(hol_flags["post_holiday"].to_numpy(dtype=bool))

    return np.c_[pre, hol, post]


def holiday_hover_template(metric_name: str) -> str:
    return (
        "<b>%{x|%Y-%m-%d}</b><br>"
        f"{metric_name}: <b>%{{y:.2f}}</b><br><br>"
        "<span style='display:inline-block; width:80px; font-weight:600;'>Pre</span>"
        "%{customdata[0]}<br>"
        "<span style='display:inline-block; width:80px; font-weight:600;'>Holiday</span>"
        "%{customdata[1]}<br>"
        "<span style='display:inline-block; width:80px; font-weight:600;'>Post</span>"
        "%{customdata[2]}<br>"
        "<extra></extra>"
    )


# ... keep everything as-is, only change THIS part in plot_decomposition_with_holidays()


def plot_decomposition_with_holidays(
    ts: pd.Series,
    hol_flags: pd.DataFrame,
    trend: pd.Series,
    seasonal: pd.Series,
    resid: pd.Series,
    title: str,
):
    """
    Hover on ALL 4 plots shows:
      - holiday_event
      - pre_holiday
      - post_holiday

    Hover is "normal" (no unified crosshair line).
    """
    idx = ts.index
    hol_flags = hol_flags.reindex(idx, fill_value=False)

    customdata = np.c_[
        hol_flags["is_holiday_or_event"].to_numpy(dtype=bool),
        hol_flags["pre_holiday"].to_numpy(dtype=bool),
        hol_flags["post_holiday"].to_numpy(dtype=bool),
    ]

    trend = trend.reindex(idx)
    seasonal = seasonal.reindex(idx)
    resid = resid.reindex(idx)

    # build colored holiday badges
    customdata = build_holiday_badges(hol_flags)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("unit_sales (dense)", "trend", "seasonal", "residual"),
    )

    fig.add_trace(
        go.Scatter(
            x=ts.index,
            y=ts.values,
            customdata=customdata,
            hovertemplate=holiday_hover_template("unit_sales"),
            name="unit_sales",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=trend.index,
            y=trend.values,
            customdata=customdata,
            hovertemplate=holiday_hover_template("trend"),
            name="trend",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=seasonal.index,
            y=seasonal.values,
            customdata=customdata,
            hovertemplate=holiday_hover_template("seasonal"),
            name="seasonal",
        ),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=resid.index,
            y=resid.values,
            mode="markers",
            customdata=customdata,
            hovertemplate=holiday_hover_template("residual"),
            name="residual",
        ),
        row=4,
        col=1,
    )

    # ✅ normal hover (no unified crosshair)
    fig.update_layout(
        height=900,
        showlegend=False,
        title=title,
        hovermode="closest",
    )
    # (Optional) explicitly disable spike lines
    fig.update_xaxes(showspikes=False)
    fig.update_yaxes(showspikes=False)

    return fig


# =====================================================
# UI
# =====================================================
st.title("🔎 Store-Item Behavior EDA")

colA, colB, colC, colD = st.columns([1, 1, 1, 1])

with colA:
    level = st.selectbox("Level", ["Daily", "Weekly"])

with colB:
    store = int(st.number_input("Store", min_value=1, value=1, step=1))

with colC:
    item = int(st.number_input("Item", min_value=1, value=1000, step=1))

with colD:
    show_sparse = st.checkbox("Also show sparse series (sales-days only)", value=False)

# =====================================================
# Select dataset settings
# =====================================================
if level == "Daily":
    df = df_daily
    df_hol = df_daily_hol

    date_col = "date"
    value_col = "unit_sales_sum"
    freq = "D"
    period = 7
    seasonal_label = "Weekly"
    season_lag = 7
else:
    df = df_weekly
    df_hol = df_weekly_hol

    date_col = "week_start" if "week_start" in df.columns else "week"
    value_col = "unit_sales_sum"
    freq = "W-MON"
    period = 52
    seasonal_label = "Yearly"
    season_lag = 52

# =====================================================
# Extract series + holiday flags (dense)
# =====================================================
ts_dense = get_series_dense(df, store, item, date_col, value_col, freq=freq)
if ts_dense.empty:
    st.warning("No data for this store-item combination")
    st.stop()

hol_dense = get_holiday_flags_dense(df_hol, store, item, date_col, freq=freq)

# Optional sparse series (only observed sales periods)
if show_sparse:
    d0 = df[(df["store_nbr"] == store) & (df["item_nbr"] == item)].copy()
    if not d0.empty:
        d0 = _normalize_time_col(d0, date_col)
        ts_sparse = d0.groupby(date_col)[value_col].sum().sort_index().astype(float)
    else:
        ts_sparse = pd.Series(dtype=float)

# =====================================================
# STL on dense series
# =====================================================
if len(ts_dense) < period * 2:
    trend = ts_dense.copy() * np.nan
    seasonal = ts_dense.copy() * np.nan
    resid = ts_dense.copy() * np.nan
    season_strength = np.nan
else:
    trend, seasonal, resid = stl_decompose(ts_dense, period)
    season_strength = seasonality_strength(ts_dense, season_lag)

# =====================================================
# Stats
# =====================================================
mean_val = float(ts_dense.mean())
std_val = float(ts_dense.std(ddof=1)) if len(ts_dense) > 1 else 0.0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Observations", f"{len(ts_dense):,}")
col2.metric("Mean", f"{mean_val:.2f}")
col3.metric("Std", f"{std_val:.2f}")
col4.metric(
    f"{seasonal_label} seasonality (ACF@{season_lag})",
    "—" if np.isnan(season_strength) else f"{season_strength:.3f}",
)
col5.metric("Zero share", f"{(ts_dense.eq(0).mean()):.1%}")

with st.expander("📊 How to interpret these time-series stats"):
    st.markdown(
        f"""
**Observations**: number of time points in the dense series (missing periods filled with 0).
**Mean**: average demand per period (including zeros).
**Std**: demand variability (including zeros). CV = Std / Mean.
**{seasonal_label} seasonality (ACF@{season_lag})**: autocorrelation at lag `{season_lag}` on the dense series.
**Zero share**: share of periods with exactly 0 demand.

**Holiday hover (all 4 panels)**:
- `holiday_event`: exact holiday
- `pre_holiday`: within 5 days before
- `post_holiday`: within 5 days after
"""
    )


# =====================================================
# Plot (hover on all panels)
# =====================================================
fig = plot_decomposition_with_holidays(
    ts_dense,
    hol_dense,
    trend,
    seasonal,
    resid,
    title=f"{level} decomposition (dense incl. zeros) · store={store} · item={item}",
)
render_plotly(fig)

if show_sparse and not ts_sparse.empty:
    st.subheader("🧾 Sparse series (sales periods only)")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(x=ts_sparse.index, y=ts_sparse.values, mode="lines+markers")
    )
    fig2.update_layout(height=300, title="Sparse series (no zero-filled periods)")
    st.plotly_chart(fig2, use_container_width=True)

with st.expander("Model interpretation"):
    st.markdown(
        """
**High seasonal component / high ACF at seasonal lag**
→ SARIMA / seasonal models likely useful

**Strong trend**
→ boosting / trend models useful

**High residual noise**
→ difficult series; consider aggregation (weekly) or stronger regularization

**High zero share + intermittent**
→ Croston-type approaches or intermittent-demand features
"""
    )
