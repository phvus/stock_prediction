"""
model/sample_runner.py  —  Standalone Demo
============================================
Run this file ALONE to prove the ARIMA model works — no database, no
frontend, no teammates needed.

It:
  1. Downloads VN-Index data from Yahoo Finance (free, public)
  2. Runs the full ARIMA pipeline
  3. Prints the forecast, evaluation, and model summary
  4. Saves the trained model to disk

HOW TO RUN:
    conda activate adatralig
    python model/sample_runner.py

EXPECTED OUTPUT:
    - 7-day forecast with confidence intervals
    - RMSE, MAE, MAPE accuracy metrics
    - Model summary (AIC, BIC, coefficients)
    - Saved model at model/saved_arima.pkl
"""

import sys
import os

# ── Make sure we can import from the project root ─────────────────────────────
# When you run `python model/sample_runner.py`, Python's working directory
# is wherever you ran the command from.  We need to add the project root
# to sys.path so that `from model import ...` works correctly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import datetime
import pandas as pd
import yfinance as yf

from model.arima_model import forecast_vnindex, check_stationarity, prepare_data
from model.arima_config import VNINDEX_TICKER, HISTORY_YEARS, DEFAULT_ORDER, FORECAST_STEPS


def fetch_sample_data() -> pd.DataFrame:
    """
    Downloads VN-Index historical data from Yahoo Finance.
    This simulates what the DB teammate would provide — a DataFrame
    with 'Date' and 'Close' columns.
    """
    print(f"📥 Downloading {VNINDEX_TICKER} data from Yahoo Finance...")
    print(f"   Period: last {HISTORY_YEARS} years")

    end_date   = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365 * HISTORY_YEARS)

    # yfinance downloads OHLCV data (Open, High, Low, Close, Volume)
    df = yf.download(VNINDEX_TICKER, start=start_date, end=end_date, progress=False)

    if df.empty:
        print("❌ Failed to download data.  Check your internet connection.")
        print("   Or try a different ticker in arima_config.py")
        sys.exit(1)

    # Reset index so 'Date' becomes a regular column (not the index)
    df.reset_index(inplace=True)

    # Handle multi-index columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(str(i) for i in col).strip('_') for col in df.columns.values]

    # Find the Close column (yfinance sometimes names it differently)
    close_col = [c for c in df.columns if 'Close' in c][0]
    date_col  = [c for c in df.columns if 'Date' in c or 'date' in c][0]

    # Rename to match our expected format
    df = df.rename(columns={close_col: "Close", date_col: "Date"})
    df = df[["Date", "Close"]]

    print(f"   ✅ Downloaded {len(df)} trading days")
    print(f"   Date range: {df['Date'].min()} → {df['Date'].max()}")
    print(f"   Price range: {df['Close'].min():.2f} → {df['Close'].max():.2f}")
    print()

    return df


def main():
    print()
    print("=" * 60)
    print("   🚀  ARIMA Model — Sample Runner / Demo")
    print("=" * 60)
    print()

    # ── Step 1: Get data (simulates DB teammate) ─────────────────────────
    df = fetch_sample_data()

    # ── Step 2: Quick stationarity check (educational) ───────────────────
    print("─" * 60)
    print("📊  STATIONARITY CHECK  (is the data suitable for ARIMA?)")
    print("─" * 60)
    series = prepare_data(df)
    stat = check_stationarity(series)
    print(f"   {stat['interpretation']}")
    print(f"   ADF statistic: {stat['adf_statistic']}")
    print(f"   p-value:       {stat['p_value']}")
    print()

    # ── Step 3: Run full pipeline ────────────────────────────────────────
    print("─" * 60)
    print(f"🔧  RUNNING ARIMA{DEFAULT_ORDER}  |  Forecast: {FORECAST_STEPS} days")
    print("─" * 60)

    result = forecast_vnindex(df, steps=FORECAST_STEPS, order=DEFAULT_ORDER, save=True)

    # ── Step 4: Print evaluation ─────────────────────────────────────────
    print()
    print("─" * 60)
    print("📈  MODEL EVALUATION  (how accurate is it?)")
    print("─" * 60)
    print(result["evaluation"]["interpretation"])

    # ── Step 5: Print forecast ───────────────────────────────────────────
    print("─" * 60)
    print("🔮  FORECAST")
    print("─" * 60)
    print(f"   Last known date:  {result['last_known_date']}")
    print(f"   Last known price: {result['last_known_price']}")
    print()
    print(f"   {'Date':<14}  {'Predicted':>10}  {'Lower CI':>10}  {'Upper CI':>10}")
    print(f"   {'─'*14}  {'─'*10}  {'─'*10}  {'─'*10}")

    fc = result["forecast"]
    for i in range(len(fc["dates"])):
        print(
            f"   {fc['dates'][i]:<14}  "
            f"{fc['predictions'][i]:>10.2f}  "
            f"{fc['lower_ci'][i]:>10.2f}  "
            f"{fc['upper_ci'][i]:>10.2f}"
        )

    # ── Step 6: Print model summary ──────────────────────────────────────
    print()
    print("─" * 60)
    print("📋  MODEL SUMMARY")
    print("─" * 60)
    print(result["model_summary"]["explanation"])

    # ── Step 7: Save confirmation ────────────────────────────────────────
    if result["saved_model_path"]:
        print(f"\n💾 Model saved to: {result['saved_model_path']}")
        print("   Your frontend teammate can load it with:")
        print("     from model import load_model, forecast")
        print("     model = load_model()")
        print("     predictions = forecast(model, steps=7)")

    print()
    print("=" * 60)
    print("   ✅  Demo complete!  Model is ready for integration.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
