import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plotly.graph_objects as go
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="VN-Index ARIMA Predictor", layout="wide", page_icon="📈")

st.title("📈 VN-Index ARIMA Forecasting")
st.markdown("""
This application demonstrates how to use an **ARIMA (AutoRegressive Integrated Moving Average)** model to forecast the VN-Index (Vietnam Stock Index).
It fetches historical data, stores it in a SQLite database, and allows you to interactively adjust the ARIMA parameters `(p, d, q)` to see how the forecast changes.
""")

DB_PATH = "data/vnindex.db"
os.makedirs("data", exist_ok=True)

# 1. Web Scraping & Database Logic
@st.cache_data(ttl=3600)
def fetch_and_store_data():
    """Fetches VN-Index data using yfinance and stores it in SQLite."""
    st.sidebar.info("Fetching fresh data from Yahoo Finance (^VNINDEX)...")
    ticker = "^VNINDEX"
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365*2) # 2 years of data
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
         st.sidebar.error("Failed to fetch data.")
         return None
         
    df.reset_index(inplace=True)
    
    # Handle multi-index columns from yfinance by flattening them if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(i) for i in col).strip('_') for col in df.columns.values]
    
    # Standardize column name (sometimes it's 'Close_^VNINDEX')
    close_col = [c for c in df.columns if 'Close' in c][0]
    df.rename(columns={close_col: 'Close'}, inplace=True)
    
    # Store in SQLite
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("vnindex_history", conn, if_exists="replace", index=False)
    conn.close()
    
    return df

def load_data_from_db():
    if not os.path.exists(DB_PATH):
        return fetch_and_store_data()
    
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM vnindex_history", conn)
        df['Date'] = pd.to_datetime(df['Date'])
        conn.close()
        return df
    except Exception:
        conn.close()
        return fetch_and_store_data()

df = load_data_from_db()

if df is not None:
    st.sidebar.success(f"Data Loaded: {len(df)} records")
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("⚙️ Model Parameters")
        st.markdown("""
        **What do these parameters mean?**
        - **p (AR - AutoRegressive)**: Lags of the dependent variable. E.g., if p=1, today's value depends on yesterday's.
        - **d (I - Integrated)**: Number of times you difference the data to make it stationary. E.g., predicting change instead of absolute value.
        - **q (MA - Moving Average)**: Lags of the forecast errors. E.g., previous shocks affecting today.
        """)
        
        p = st.number_input("p (AR Terms)", min_value=0, max_value=5, value=1)
        d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
        q = st.number_input("q (MA Terms)", min_value=0, max_value=5, value=1)
        
        forecast_steps = st.slider("Days to Forecast", min_value=1, max_value=30, value=7)
        
    with col2:
        st.subheader("📊 Forecast Visualization")
        
        # Prepare data (handle missing values/weekends)
        df.dropna(subset=['Close'], inplace=True)
        # Convert Close column to float in case it isn't
        df['Close'] = df['Close'].astype(float)
        
        df_model = df.set_index('Date')['Close']
        
        if len(df_model) > 10:
            with st.spinner("Training ARIMA Model..."):
                try:
                    model = ARIMA(df_model, order=(p, d, q))
                    results = model.fit()
                    
                    # Forecast
                    forecast = results.forecast(steps=forecast_steps)
                    
                    # Create future dates
                    last_date = df['Date'].max()
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')
                    
                    # Plotly Chart
                    fig = go.Figure()
                    
                    # Historical (Plot last 100 days for clarity)
                    fig.add_trace(go.Scatter(x=df['Date'][-100:], y=df['Close'][-100:], mode='lines', name='Historical', line=dict(color='#00d4ff', width=2)))
                    
                    # Forecast
                    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast', line=dict(color='#ff007f', dash='dash', width=3)))
                    
                    fig.update_layout(
                        title="VN-Index History & ARIMA Forecast",
                        xaxis_title="Date",
                        yaxis_title="Index Value",
                        height=500,
                        template="plotly_dark",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Show Model Summary"):
                        st.text(results.summary().as_text())
                        
                except Exception as e:
                    st.error(f"Error fitting model with parameters ({p}, {d}, {q}). Try different parameters.\nDetails: {e}")

    st.markdown("---")
    st.markdown("### 📝 Educational Notes")
    st.markdown("""
    In the context of the **Adaptive Traffic Light Project**, predicting traffic flow is conceptually identical to predicting stock indexes. Both represent **Time Series** data.
    
    By identifying patterns (seasonality, trends, rolling averages) in past traffic volume, an ARIMA model can accurately forecast how many cars will arrive at an intersection in the next 10 minutes. 
    This allows the traffic light system to proactively adjust green light durations _before_ the congestion even happens, rather than simply reacting to sensors.
    """)
