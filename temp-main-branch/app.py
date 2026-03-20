import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    forecast_arimax,
    load_companies,
    load_sector_history,
    load_symbol_history,
    summarize_horizons,
)

st.set_page_config(page_title="Stock Forecast Dashboard", page_icon="📈", layout="wide")
st.title("📈 PostgreSQL Stock Forecast Dashboard (ARIMAX)")
st.caption("Offline-ready demo included. Select company or sector group, then forecast 1/3/7-day trend.")

DEMO_COMPANIES = pd.DataFrame(
    [
        {"symbol": "AAPL", "sector": "Technology"},
        {"symbol": "MSFT", "sector": "Technology"},
        {"symbol": "JPM", "sector": "Financial Services"},
        {"symbol": "BAC", "sector": "Financial Services"},
        {"symbol": "XOM", "sector": "Energy"},
    ]
)


def _stable_seed(symbol: str) -> int:
    digest = hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16)


def build_demo_history(symbol: str, years: int = 2) -> pd.DataFrame:
    periods = years * 252
    seed = _stable_seed(symbol)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=periods, freq="B")
    drift = (seed % 9 + 2) / 100
    noise = rng.normal(0, 1.2, periods)
    seasonal = 2.3 * np.sin(np.arange(periods) / 7)
    close = 90 + np.arange(periods) * drift + seasonal + np.cumsum(noise) * 0.1
    return pd.DataFrame({"Date": dates, "Close": close.round(2)})


with st.sidebar:
    st.subheader("⚙️ Forecast Settings")
    mode = st.radio("Target", ["Company", "Sector Group"], index=0)
    selected_horizons = st.multiselect("Prediction days", [1, 3, 7], default=[1, 3, 7])
    years = st.slider("Lookback years", min_value=1, max_value=5, value=2)
    auto_arima = st.checkbox("Auto ARIMAX order search", value=True)
    if auto_arima:
        p, d, q = 1, 1, 1
        st.caption("Manual (p,d,q) disabled because auto-search is on.")
    else:
        p = st.number_input("p", min_value=0, max_value=5, value=1)
        d = st.number_input("d", min_value=0, max_value=2, value=1)
        q = st.number_input("q", min_value=0, max_value=5, value=1)

use_demo = False
try:
    companies = load_companies()
except Exception as exc:
    use_demo = True
    companies = DEMO_COMPANIES.copy()
    st.warning(f"Database unavailable ({exc}). Running deterministic demo mode.")

if companies.empty:
    st.error("No company metadata found. Please import stock.sql / populate company_info first.")
    st.stop()

sectors = sorted(companies["sector"].dropna().unique().tolist())

if mode == "Company":
    c1, c2 = st.columns([1, 2])
    with c1:
        sector_filter = st.selectbox("Sector filter", ["All"] + sectors)
    with c2:
        symbols = companies["symbol"].tolist() if sector_filter == "All" else companies.loc[companies["sector"] == sector_filter, "symbol"].tolist()
        symbol = st.selectbox("Company symbol", symbols)
    target_df = build_demo_history(symbol, years) if use_demo else load_symbol_history(symbol, years)
    title = f"{symbol} - historical vs forecast"
else:
    sector = st.selectbox("Sector group", sectors)
    if use_demo:
        sector_symbols = companies.loc[companies["sector"] == sector, "symbol"].tolist()
        group_data = [build_demo_history(s, years).assign(Symbol=s) for s in sector_symbols]
        target_df = pd.concat(group_data, ignore_index=True).groupby("Date", as_index=False)["Close"].mean().sort_values("Date")
        failed = []
    else:
        target_df, failed = load_sector_history(companies, sector, years)
    title = f"{sector} sector mean close - historical vs forecast"
    if failed:
        st.info(f"Skipped {len(failed)} symbols with missing tables/data: {', '.join(failed[:8])}")

if target_df.empty:
    st.error("No history for selected target.")
    st.stop()
if not selected_horizons:
    st.error("Select at least one horizon (1, 3, or 7 days).")
    st.stop()

with st.spinner("Training ARIMAX and forecasting..."):
    result = forecast_arimax(
        target_df,
        steps=7,
        auto_order=auto_arima,
        manual_order=(int(p), int(d), int(q)),
    )

summary = summarize_horizons(result, selected_horizons)
st.subheader("Forecast summary")
st.dataframe(summary, use_container_width=True)

hist = target_df.copy()
hist["Date"] = pd.to_datetime(hist["Date"])
hist = hist.sort_values("Date")
f_dates = pd.to_datetime(result["forecast"]["dates"])

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=hist["Date"].tail(180),
        y=hist["Close"].tail(180),
        mode="lines",
        name="Historical",
        line=dict(color="#00B5E2", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=f_dates,
        y=result["forecast"]["predictions"],
        mode="lines+markers",
        name="Forecast",
        line=dict(color="#D81B60", width=3, dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=f_dates,
        y=result["forecast"]["upper_95"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=f_dates,
        y=result["forecast"]["lower_95"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(216,27,96,0.15)",
        line=dict(width=0),
        name="95% confidence",
    )
)
fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", template="plotly_dark", height=560)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Beginner explanation"):
    st.markdown(
        """
- **What ARIMAX does:** it learns from past prices and extra factors (here: weekday/month/trend).
- **What Auto ARIMAX does:** it tests many `(p,d,q)` combinations and keeps the one with the best AIC score.
- **How trend labels are made:** compare the predicted price to the latest real price.
  - `⬆️ Upward` if rise > 0.2%
  - `⬇️ Downward` if drop < -0.2%
  - `➡️ Sideways` otherwise
        """
    )

with st.expander("Debug panel"):
    st.write({
        "chosen_order": result["chosen_order"],
        "aic": result["debug"]["aic"],
        "bic": result["debug"]["bic"],
        "num_history_rows": result["debug"]["num_history_rows"],
        "note": result["debug"]["note"],
    })
    trials = pd.DataFrame(result["debug"]["order_trials"])
    st.dataframe(trials, use_container_width=True)
    st.caption("Detailed logs are saved to app_debug.log")
