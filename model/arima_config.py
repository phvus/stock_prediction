"""
model/arima_config.py  —  Configuration Constants
===================================================
All tunable parameters in ONE place.

WHY:  Instead of scattering magic numbers inside the model code, we put them
      here so anyone (you, your teammates, or the judges) can instantly see
      what dials are available and tweak them without digging into the logic.

HOW TEAMMATES USE THIS:
      from model.arima_config import DEFAULT_ORDER, FORECAST_STEPS
"""

# ──────────────────────────────────────────────────────────────────────────────
# ARIMA ORDER  (p, d, q)
# ──────────────────────────────────────────────────────────────────────────────
# p = number of AutoRegressive terms   (how many past values influence today)
# d = differencing order               (how many times we subtract consecutive
#                                        values to remove trend)
# q = number of Moving Average terms   (how many past forecast errors matter)
#
# (1, 1, 1) is a safe, classic starting point for financial time series:
#   - p=1 : today depends on yesterday
#   - d=1 : we look at *changes* rather than raw prices (removes trend)
#   - q=1 : we correct based on yesterday's error
DEFAULT_ORDER = (1, 1, 1)

# ──────────────────────────────────────────────────────────────────────────────
# FORECAST HORIZON
# ──────────────────────────────────────────────────────────────────────────────
# How many business days into the future to predict.
# 7 ≈ about 1.5 calendar weeks (stock markets are closed on weekends).
FORECAST_STEPS = 7

# ──────────────────────────────────────────────────────────────────────────────
# DATA REQUIREMENTS
# ──────────────────────────────────────────────────────────────────────────────
# Minimum number of data points (rows) needed to fit ARIMA reliably.
# With fewer than this, the model may produce garbage or fail to converge.
# 60 trading days ≈ roughly 3 months of data.
MIN_DATA_POINTS = 60

# ──────────────────────────────────────────────────────────────────────────────
# COLUMN NAMES  —  The contract with your DB teammate
# ──────────────────────────────────────────────────────────────────────────────
# Your database teammate must provide a pandas DataFrame with AT LEAST these
# two columns.  The names MUST MATCH exactly (case-sensitive).
#
#   DATE_COL  →  the trading date    (dtype: str or datetime)
#   CLOSE_COL →  the closing price   (dtype: float)
#
# Example DataFrame your DB teammate should provide:
#
#   |   Date       |  Close   |
#   |------------- |----------|
#   |  2024-01-02  |  1130.5  |
#   |  2024-01-03  |  1135.2  |
#   |  ...         |  ...     |
DATE_COL  = "Date"
CLOSE_COL = "Close"

# ──────────────────────────────────────────────────────────────────────────────
# STATIONARITY TEST
# ──────────────────────────────────────────────────────────────────────────────
# Significance level for the Augmented Dickey-Fuller (ADF) test.
# If the p-value from ADF is BELOW this threshold, we consider the data
# "stationary" (no trend / constant mean & variance over time).
# 0.05 = 5% ⇒ 95% confidence.
ADF_SIGNIFICANCE = 0.05

# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────────
# When evaluating the model, we hold back the last TEST_SIZE_RATIO of the data
# as "test" data so we can measure how well the model predicts unseen values.
# 0.2 = 80% train, 20% test.
TEST_SIZE_RATIO = 0.2

# ──────────────────────────────────────────────────────────────────────────────
# CONFIDENCE INTERVAL
# ──────────────────────────────────────────────────────────────────────────────
# When forecasting, ARIMA can give a range: "the price will be between X and Y
# with Z% confidence."  0.95 ⇒ 95% confidence interval.
CONFIDENCE_LEVEL = 0.95

# ──────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────────
# Default path to save/load the trained model.
# Your frontend teammate can load this file to avoid re-training every time.
SAVED_MODEL_PATH = "model/saved_arima.pkl"

# ──────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE TICKER
# ──────────────────────────────────────────────────────────────────────────────
# This is the ticker symbol Yahoo Finance uses for the sample data.
# Note: Yahoo Finance sometimes stops providing Vietnamese tickers like '^VNINDEX'. 
# We use 'AAPL' (Apple) as a reliable sample for the demo runner so you can test it.
# Your DB teammate will use real VN-Index data from their own source.
VNINDEX_TICKER = "AAPL"

# How many years of historical data to download for testing.
HISTORY_YEARS = 2
