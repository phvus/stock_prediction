import datetime
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
from psycopg2 import sql
import streamlit as st

from model import forecast_vnindex

st.set_page_config(page_title="Stock Predictor (PostgreSQL + ARIMA)", layout="wide", page_icon="📈")
st.title("📈 Stock Prediction Dashboard")
st.markdown(
    "Predict 1-day, 3-day, and 7-day price trends using ARIMA from PostgreSQL data. "
    "You can forecast for a single company or a sector group."
)

TREND_THRESHOLD = 0.2
DRIFT_BY_SYMBOL = {
    "AAPL": 0.03,
    "MSFT": 0.03,
    "JPM": 0.015,
    "BAC": 0.015,
    "XOM": 0.005,
}


def _get_db_config() -> dict:
    return {
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "stock"),
    }


def _safe_table_name(symbol: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", symbol):
        raise ValueError(f"Invalid symbol/table name: {symbol}")
    return symbol.lower()


def _build_demo_companies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "sector": "Technology"},
            {"symbol": "MSFT", "sector": "Technology"},
            {"symbol": "JPM", "sector": "Financial Services"},
            {"symbol": "BAC", "sector": "Financial Services"},
            {"symbol": "XOM", "sector": "Energy"},
        ]
    )


def _build_demo_history(symbol: str, years: int = 2) -> pd.DataFrame:
    seed = abs(hash(symbol)) % (2**32)
    rng = pd.Series(range(years * 252), dtype=float)
    drift = DRIFT_BY_SYMBOL.get(symbol, 0.01)
    base = 100 + (seed % 50)
    close = base + (rng * drift) + (pd.Series(np.sin(rng / 8.0)) * 2.5)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=len(rng), freq="B")
    return pd.DataFrame({"Date": dates, "Close": close.round(2)})


@st.cache_data(ttl=300)
def load_companies() -> pd.DataFrame:
    conn = psycopg2.connect(**_get_db_config())
    try:
        query = """
        SELECT symbol, COALESCE(icb_name2, 'Unknown') AS sector
        FROM public.company_info
        ORDER BY symbol
        """
        companies = pd.read_sql(query, conn)
        return companies
    finally:
        conn.close()


def load_symbol_history(symbol: str, years: int = 2) -> pd.DataFrame:
    table = _safe_table_name(symbol)
    start_date = datetime.date.today() - datetime.timedelta(days=365 * years)
    conn = psycopg2.connect(**_get_db_config())
    try:
        query = sql.SQL(
            "SELECT time AS \"Date\", close AS \"Close\" "
            "FROM {} WHERE time >= %s ORDER BY time"
        ).format(sql.Identifier(table))
        with conn.cursor() as cur:
            cur.execute(query, (start_date,))
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=["Date", "Close"])
    finally:
        conn.close()


def load_sector_history(companies: pd.DataFrame, sector: str, years: int = 2) -> pd.DataFrame:
    symbols = companies[companies["sector"] == sector]["symbol"].tolist()
    all_histories = []
    failed_symbols = []
    for symbol in symbols:
        try:
            history = load_symbol_history(symbol, years=years)
        except Exception:
            failed_symbols.append(symbol)
            continue
        if not history.empty:
            history["Symbol"] = symbol
            all_histories.append(history)

    if failed_symbols:
        st.warning(
            f"Skipped {len(failed_symbols)} symbols with missing/unavailable history: "
            f"{', '.join(failed_symbols[:5])}"
        )

    if not all_histories:
        return pd.DataFrame(columns=["Date", "Close"])

    group_df = pd.concat(all_histories, ignore_index=True)
    group_df["Date"] = pd.to_datetime(group_df["Date"])
    grouped = group_df.groupby("Date", as_index=False)["Close"].mean()
    return grouped.sort_values("Date")


def classify_trend(last_price: float, predicted_price: float) -> str:
    change_pct = ((predicted_price - last_price) / last_price) * 100
    if change_pct > TREND_THRESHOLD:
        return "⬆️ Upward"
    if change_pct < -TREND_THRESHOLD:
        return "⬇️ Downward"
    return "➡️ Sideways"


def summarize_horizons(result: dict, selected_horizons: list[int]) -> pd.DataFrame:
    last_price = float(result["last_known_price"])
    dates = result["forecast"]["dates"]
    preds = result["forecast"]["predictions"]
    rows = []
    for days in selected_horizons:
        idx = days - 1
        predicted_price = float(preds[idx])
        rows.append(
            {
                "Horizon": f"{days} day" if days == 1 else f"{days} days",
                "Forecast date": dates[idx],
                "Predicted close": round(predicted_price, 2),
                "Change (%)": round(((predicted_price - last_price) / last_price) * 100, 2),
                "Trend": classify_trend(last_price, predicted_price),
            }
        )
    return pd.DataFrame(rows)


with st.sidebar:
    st.subheader("⚙️ Settings")
    mode = st.radio("Prediction target", ["Company", "Sector Group"], index=0)
    p = st.number_input("p", min_value=0, max_value=5, value=1)
    d = st.number_input("d", min_value=0, max_value=2, value=1)
    q = st.number_input("q", min_value=0, max_value=5, value=1)
    selected_horizons = st.multiselect(
        "Select prediction days",
        options=[1, 3, 7],
        default=[1, 3, 7],
    )
    years = st.slider("Lookback years", min_value=1, max_value=5, value=2)

use_demo_data = False
try:
    companies_df = load_companies()
except Exception as exc:
    st.warning(f"PostgreSQL unavailable ({exc}). Running in demo mode with synthetic data.")
    companies_df = _build_demo_companies()
    use_demo_data = True

if companies_df.empty:
    st.warning("No companies found in public.company_info. Please import data first.")
    st.stop()

sectors = sorted(companies_df["sector"].dropna().unique().tolist())

if mode == "Company":
    col_sel_1, col_sel_2 = st.columns([1, 2])
    with col_sel_1:
        selected_sector = st.selectbox("Sector filter", options=["All"] + sectors, index=0)
    with col_sel_2:
        if selected_sector == "All":
            symbols = companies_df["symbol"].tolist()
        else:
            symbols = companies_df[companies_df["sector"] == selected_sector]["symbol"].tolist()
        selected_symbol = st.selectbox("Company symbol", options=symbols)
    target_df = _build_demo_history(selected_symbol, years=years) if use_demo_data else load_symbol_history(selected_symbol, years=years)
    chart_title = f"{selected_symbol} — Historical + Forecast"
else:
    selected_sector = st.selectbox("Sector group", options=sectors)
    if use_demo_data:
        symbol_list = companies_df[companies_df["sector"] == selected_sector]["symbol"].tolist()
        sector_dfs = [_build_demo_history(symbol, years=years).assign(Symbol=symbol) for symbol in symbol_list]
        all_df = pd.concat(sector_dfs, ignore_index=True)
        target_df = all_df.groupby("Date", as_index=False)["Close"].mean().sort_values("Date")
    else:
        target_df = load_sector_history(companies_df, selected_sector, years=years)
    chart_title = f"{selected_sector} Sector (mean close) — Historical + Forecast"

if target_df.empty:
    st.warning("No historical data found for the selected target.")
    st.stop()

if not selected_horizons:
    st.warning("Please select at least one prediction horizon.")
    st.stop()

with st.spinner("Training ARIMA model and generating forecasts..."):
    model_result = forecast_vnindex(
        df=target_df,
        steps=7,
        order=(int(p), int(d), int(q)),
        save=False,
    )

summary_df = summarize_horizons(model_result, selected_horizons=sorted(selected_horizons))
st.subheader("📌 1/3/7 Day Prediction Summary")
st.dataframe(summary_df, use_container_width=True)

history_df = target_df.copy()
history_df["Date"] = pd.to_datetime(history_df["Date"])
history_df = history_df.sort_values("Date")

forecast_dates = pd.to_datetime(model_result["forecast"]["dates"])
forecast_values = model_result["forecast"]["predictions"]

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_df["Date"].tail(120),
        y=history_df["Close"].tail(120),
        mode="lines",
        name="Historical",
        line=dict(color="#00d4ff", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name="Forecast (7-day)",
        line=dict(color="#ff007f", width=3, dash="dash"),
    )
)
fig.update_layout(
    title=chart_title,
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=520,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Model diagnostics"):
    st.text(model_result["evaluation"]["interpretation"])
    st.text(model_result["model_summary"]["explanation"])
