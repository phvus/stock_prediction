"""
model/arima_model.py  —  The ARIMA Model Engine
=================================================
This is YOUR part of the project.  Every function is the building block
that your teammates (Database person & Frontend person) will call.

WHAT THIS FILE DOES:
  1. Takes historical VN-Index closing prices (a pandas DataFrame)
  2. Validates and cleans the data
  3. Tests whether the data is "stationary" (required for ARIMA)
  4. Fits an ARIMA model with a given (p, d, q) order
  5. Produces a forecast (future prices + confidence intervals)
  6. Evaluates accuracy (RMSE, MAE, MAPE)
  7. Saves / loads the trained model to/from disk

EVERY SINGLE LINE is commented so you can explain it to the judges.

HOW TEAMMATES USE THIS:
  from model.arima_model import prepare_data, fit_arima, forecast, evaluate_model
"""

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS — libraries we need
# ══════════════════════════════════════════════════════════════════════════════

import warnings                          # To suppress non-critical warnings that clutter output
warnings.filterwarnings("ignore")        # ARIMA sometimes throws convergence warnings; we silence them

import numpy as np                       # Numerical operations (arrays, math). The backbone of all ML.
import pandas as pd                      # DataFrames — the "spreadsheet" library of Python
import joblib                            # Save/load Python objects to disk (our trained model)
import os                                # File & directory operations (check if file exists, etc.)

from statsmodels.tsa.arima.model import ARIMA          # The ARIMA model itself
from statsmodels.tsa.stattools import adfuller          # Augmented Dickey-Fuller test (checks stationarity)
from sklearn.metrics import (                           # Standard ML evaluation metrics
    mean_squared_error,                                 #   RMSE — average error in same units as price
    mean_absolute_error,                                #   MAE  — average absolute error
)

# Our configuration constants (default order, column names, etc.)
from model.arima_config import (
    DEFAULT_ORDER,
    FORECAST_STEPS,
    MIN_DATA_POINTS,
    DATE_COL,
    CLOSE_COL,
    ADF_SIGNIFICANCE,
    TEST_SIZE_RATIO,
    CONFIDENCE_LEVEL,
    SAVED_MODEL_PATH,
)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 1:  prepare_data
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Take a raw DataFrame from the DB teammate and make it ready
#           for ARIMA.  This is the "cleaning" step.
#
# WHY:  Real-world data is messy.  Dates might be strings, there might be
#       missing rows (holidays), or NaN values.  ARIMA needs a clean,
#       sorted, numeric time series.
# ──────────────────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> pd.Series:
    """
    Cleans a raw DataFrame and returns a pandas Series of closing prices
    indexed by date, ready for ARIMA.

    Parameters:
        df : pd.DataFrame  — must have columns from arima_config:
                              DATE_COL ("Date") and CLOSE_COL ("Close")

    Returns:
        pd.Series with DatetimeIndex and float values (closing prices)

    Raises:
        ValueError if data is missing required columns or too short
    """

    # ── Step 1: Validate that the expected columns exist ──────────────────
    # If the DB teammate sends a DataFrame without 'Date' or 'Close',
    # we want to FAIL LOUDLY with a clear error message, not silently
    # produce garbage.
    if DATE_COL not in df.columns:
        raise ValueError(
            f"Column '{DATE_COL}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}.  "
            f"Tell your DB teammate to include a '{DATE_COL}' column."
        )
    if CLOSE_COL not in df.columns:
        raise ValueError(
            f"Column '{CLOSE_COL}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}.  "
            f"Tell your DB teammate to include a '{CLOSE_COL}' column."
        )

    # ── Step 2: Make a copy so we don't mess up the original ──────────────
    # In Python, DataFrames are passed by reference.  If we modify `df`
    # directly, the caller's original data would change too.  .copy()
    # prevents that.
    data = df[[DATE_COL, CLOSE_COL]].copy()

    # ── Step 3: Convert the Date column to proper datetime objects ─────────
    # Dates might come as strings like "2024-01-15" from the database.
    # pd.to_datetime converts them into datetime objects so pandas can
    # sort them and compute date ranges.
    data[DATE_COL] = pd.to_datetime(data[DATE_COL])

    # ── Step 4: Sort by date (oldest first) ───────────────────────────────
    # ARIMA expects chronological order.  If data comes from a database
    # query, the order is not guaranteed, so we always sort.
    data = data.sort_values(DATE_COL)

    # ── Step 5: Set the date as the index ─────────────────────────────────
    # ARIMA in statsmodels works best when the time column IS the index
    # of the Series, not a regular column.  This also allows us to use
    # date-based slicing like series["2024-01":"2024-06"].
    data = data.set_index(DATE_COL)

    # ── Step 6: Convert the Close column to numeric (float) ───────────────
    # Sometimes database exports have prices as strings like "1,234.50".
    # pd.to_numeric forces them into numbers; 'coerce' turns unparseable
    # values into NaN (which we handle in the next step).
    data[CLOSE_COL] = pd.to_numeric(data[CLOSE_COL], errors="coerce")

    # ── Step 7: Drop any rows with NaN values ─────────────────────────────
    # NaN (Not a Number) = missing data.  ARIMA cannot handle gaps.
    # We drop these rows entirely.  In stock data, this usually happens
    # on holidays when the market was closed.
    data = data.dropna()

    # ── Step 8: Check we have enough data ─────────────────────────────────
    # ARIMA needs a minimum amount of history to find patterns.
    # With only 10 data points, it can't reliably learn anything.
    if len(data) < MIN_DATA_POINTS:
        raise ValueError(
            f"Not enough data: got {len(data)} rows, need at least "
            f"{MIN_DATA_POINTS}.  Ask your DB teammate for more history."
        )

    # ── Step 9: Extract as a Series and set a business-day frequency ──────
    # ARIMA in statsmodels likes to know the "frequency" of the data.
    # 'B' = business days (Mon–Fri).  asfreq('B') fills the index with
    # every business day; ffill() copies the previous day's price for
    # any missing dates (e.g., a random holiday).
    series = data[CLOSE_COL].asfreq("B")    # align to business-day calendar
    series = series.ffill()                  # forward-fill any gaps

    # ── Step 10: Name the series (cosmetic, but helps in debug output) ────
    series.name = "VN-Index Close"

    return series


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 2:  check_stationarity
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Test whether the time series is "stationary."
#
# WHAT IS STATIONARITY?
#   A series is stationary if its mean, variance, and autocorrelation
#   structure DO NOT CHANGE over time.
#   - Stock prices are NOT stationary (they trend up/down over years).
#   - Stock *returns* (daily % change) usually ARE stationary.
#   ARIMA's "I" (Integrated) part handles this by differencing.
#
# WHY IT MATTERS:
#   ARIMA assumes stationarity.  If you feed it non-stationary data,
#   the forecast will be unreliable.  The ADF test tells us whether
#   we need differencing (d > 0) or not.
# ──────────────────────────────────────────────────────────────────────────────

def check_stationarity(series: pd.Series) -> dict:
    """
    Runs the Augmented Dickey-Fuller (ADF) test on the series.

    Returns:
        dict with keys:
          - "is_stationary" : bool   — True if p-value < ADF_SIGNIFICANCE
          - "adf_statistic" : float  — the test statistic (more negative = more stationary)
          - "p_value"       : float  — probability this is NOT stationary
          - "critical_values": dict  — thresholds at 1%, 5%, 10% significance
          - "interpretation": str    — plain-English explanation
    """

    # Run the ADF test.  It returns a tuple of values.
    # We unpack the ones we need:
    #   result[0] = ADF test statistic
    #   result[1] = p-value
    #   result[4] = critical values dict {'1%': ..., '5%': ..., '10%': ...}
    result = adfuller(
        series.dropna(),    # drop any NaN just in case
        autolag="AIC",      # automatically pick the best lag length using AIC
    )

    adf_stat    = result[0]   # The test statistic — more negative = more evidence of stationarity
    p_value     = result[1]   # The probability that the data is NOT stationary
    crit_values = result[4]   # Critical values at different confidence levels

    # DECISION RULE:
    # If p-value < 0.05 (our ADF_SIGNIFICANCE), we REJECT the null hypothesis
    # (which says "the data has a unit root = is NOT stationary").
    # Rejecting it means: "the data IS stationary."
    is_stationary = p_value < ADF_SIGNIFICANCE

    # Build a plain-English interpretation for the judges / debug output
    if is_stationary:
        interpretation = (
            f"✅ STATIONARY (p-value={p_value:.4f} < {ADF_SIGNIFICANCE}).  "
            f"The data has no significant trend.  ARIMA can work directly on it."
        )
    else:
        interpretation = (
            f"⚠️ NON-STATIONARY (p-value={p_value:.4f} ≥ {ADF_SIGNIFICANCE}).  "
            f"The data has a trend.  ARIMA will difference it (d≥1) to remove "
            f"the trend before modelling."
        )

    return {
        "is_stationary":   is_stationary,
        "adf_statistic":   round(adf_stat, 4),
        "p_value":         round(p_value, 6),
        "critical_values": {k: round(v, 4) for k, v in crit_values.items()},
        "interpretation":  interpretation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 3:  fit_arima
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Train the ARIMA model on the time series.
#
# ANALOGY:  Imagine teaching someone to predict tomorrow's temperature by
#           showing them the last 2 years of weather data.  "fit" = the
#           learning process where the model finds the best coefficients
#           (weights) to describe the pattern in the data.
# ──────────────────────────────────────────────────────────────────────────────

def fit_arima(series: pd.Series, order: tuple = None):
    """
    Fits an ARIMA model to the given time series.

    Parameters:
        series : pd.Series   — cleaned time series from prepare_data()
        order  : tuple(p,d,q) — ARIMA order.  Defaults to DEFAULT_ORDER.

    Returns:
        results : ARIMAResultsWrapper — the fitted model object
                  (contains coefficients, residuals, AIC, etc.)
    """

    # Use default order if none specified
    if order is None:
        order = DEFAULT_ORDER

    # ── Create the ARIMA model object ─────────────────────────────────────
    # This DOES NOT train the model yet.  It just sets up the mathematical
    # structure:  "I want an ARIMA with p={order[0]}, d={order[1]}, q={order[2]}
    # applied to this particular time series."
    model = ARIMA(
        series,        # the historical data to learn from
        order=order,   # (p, d, q) — the three hyperparameters
    )

    # ── Fit (train) the model ─────────────────────────────────────────────
    # .fit() is where the actual math happens.  The model uses Maximum
    # Likelihood Estimation (MLE) to find the AR and MA coefficients that
    # best explain the patterns in the data.
    #
    # Think of it like finding the "best line" in linear regression,
    # but for a time series instead of a scatter plot.
    results = model.fit()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 4:  forecast
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Given a trained model, produce future predictions.
#
# WHAT YOU GET BACK:
#   - The predicted values (e.g., "VN-Index will be 1235.5 tomorrow")
#   - A confidence interval ("we're 95% sure it'll be between 1210 and 1260")
# ──────────────────────────────────────────────────────────────────────────────

def forecast(results, steps: int = None) -> dict:
    """
    Generates future predictions from a fitted ARIMA model.

    Parameters:
        results : fitted ARIMA model (from fit_arima)
        steps   : how many business days to predict (default: FORECAST_STEPS)

    Returns:
        dict with keys:
          - "dates"       : list of future dates (as strings "YYYY-MM-DD")
          - "predictions" : list of predicted closing prices
          - "lower_ci"    : list of lower bounds (95% confidence)
          - "upper_ci"    : list of upper bounds (95% confidence)
    """

    if steps is None:
        steps = FORECAST_STEPS

    # ── Get the forecast ──────────────────────────────────────────────────
    # .get_forecast(steps) returns a ForecastResults object that contains
    # both the point predictions AND the confidence intervals.
    forecast_result = results.get_forecast(steps=steps)

    # ── Extract point predictions ─────────────────────────────────────────
    # .predicted_mean is a pandas Series of the forecasted values.
    # Example: [1230.5, 1232.1, 1228.3, ...]
    predicted_values = forecast_result.predicted_mean

    # ── Extract confidence intervals ──────────────────────────────────────
    # .conf_int(alpha) returns a DataFrame with two columns:
    #   "lower Close"  and  "upper Close"
    # alpha = 1 - CONFIDENCE_LEVEL.  If CONFIDENCE_LEVEL=0.95, alpha=0.05,
    # meaning there's a 5% chance the true value falls outside this range.
    alpha = 1 - CONFIDENCE_LEVEL
    confidence_intervals = forecast_result.conf_int(alpha=alpha)

    # ── Build the future dates ────────────────────────────────────────────
    # We need to know WHICH dates these predictions correspond to.
    # predicted_values already has a DatetimeIndex from the model, so we
    # just extract it.
    future_dates = predicted_values.index

    # ── Package everything into a clean dictionary ────────────────────────
    # This is what the frontend teammate receives.
    return {
        "dates":       [d.strftime("%Y-%m-%d") for d in future_dates],
        "predictions": [round(float(v), 2)     for v in predicted_values],
        "lower_ci":    [round(float(v), 2)     for v in confidence_intervals.iloc[:, 0]],
        "upper_ci":    [round(float(v), 2)     for v in confidence_intervals.iloc[:, 1]],
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 5:  evaluate_model
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Measure how accurate the model is.
#
# HOW:  We split the data into TRAIN (80%) and TEST (20%).
#       Train the model on the first 80%, then forecast the last 20%.
#       Compare forecasted values vs actual values.
#
# METRICS EXPLAINED:
#   RMSE (Root Mean Squared Error):
#       "On average, our prediction is off by X index points."
#       Lower = better.  Example: RMSE=15 means average error of 15 points.
#
#   MAE (Mean Absolute Error):
#       Same idea as RMSE but doesn't penalize large errors as heavily.
#
#   MAPE (Mean Absolute Percentage Error):
#       "On average, our prediction is off by X%."
#       MAPE=1.5 means average error of 1.5%.  Very intuitive for judges.
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(series: pd.Series, order: tuple = None) -> dict:
    """
    Performs a train/test split, trains ARIMA on the training set,
    forecasts the test period, and measures accuracy.

    Parameters:
        series : pd.Series   — cleaned time series
        order  : tuple(p,d,q) — ARIMA order (default: DEFAULT_ORDER)

    Returns:
        dict with keys:
          - "rmse"          : float — Root Mean Squared Error
          - "mae"           : float — Mean Absolute Error
          - "mape"          : float — Mean Absolute Percentage Error (%)
          - "train_size"    : int
          - "test_size"     : int
          - "actual"        : list of actual test values
          - "predicted"     : list of predicted test values
          - "interpretation": str — plain-English summary
    """

    if order is None:
        order = DEFAULT_ORDER

    # ── Split the data ────────────────────────────────────────────────────
    # Example: 500 data points → train on first 400, test on last 100.
    split_point = int(len(series) * (1 - TEST_SIZE_RATIO))
    train = series[:split_point]   # first 80%
    test  = series[split_point:]   # last 20%

    # ── Train on the training set ─────────────────────────────────────────
    model   = ARIMA(train, order=order)
    results = model.fit()

    # ── Forecast for the test period ──────────────────────────────────────
    # We ask the model to predict exactly len(test) steps ahead.
    predictions = results.forecast(steps=len(test))

    # ── Align predictions with actual values ──────────────────────────────
    # Convert both to numpy arrays to ensure we're comparing apples to apples.
    actual    = test.values                        # real prices during test period
    predicted = predictions.values[:len(actual)]   # model's guesses

    # ── Calculate metrics ─────────────────────────────────────────────────

    # RMSE: sqrt( average( (actual - predicted)² ) )
    # Squares the errors first (so big errors are penalized more),
    # takes the average, then takes the square root to get back to
    # the original units (index points).
    rmse_val = float(np.sqrt(mean_squared_error(actual, predicted)))

    # MAE: average( |actual - predicted| )
    # Simpler than RMSE — just the average of absolute differences.
    mae_val = float(mean_absolute_error(actual, predicted))

    # MAPE: average( |actual - predicted| / |actual| ) × 100
    # Expresses error as a percentage.  Very useful for judges because
    # "1.5% error" is more intuitive than "18.3 index points error."
    mape_val = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    # ── Human-readable interpretation ─────────────────────────────────────
    interpretation = (
        f"Model ARIMA{order} — Evaluation on {len(test)} test days:\n"
        f"  • RMSE = {rmse_val:.2f} index points  "
        f"(average error in same units as VN-Index)\n"
        f"  • MAE  = {mae_val:.2f} index points  "
        f"(average absolute difference)\n"
        f"  • MAPE = {mape_val:.2f}%  "
        f"(average percentage error — lower is better)\n"
    )

    if mape_val < 2:
        interpretation += "  → 🟢 Excellent accuracy (<2% error)\n"
    elif mape_val < 5:
        interpretation += "  → 🟡 Good accuracy (2-5% error)\n"
    else:
        interpretation += "  → 🔴 Consider tuning parameters (>5% error)\n"

    return {
        "rmse":           round(rmse_val, 2),
        "mae":            round(mae_val, 2),
        "mape":           round(mape_val, 2),
        "train_size":     len(train),
        "test_size":      len(test),
        "actual":         [round(float(v), 2) for v in actual],
        "predicted":      [round(float(v), 2) for v in predicted],
        "interpretation": interpretation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 6:  get_model_summary
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Extract a human-readable summary of the trained model.
#
# This is what you show the judges when they ask "what did the model learn?"
# ──────────────────────────────────────────────────────────────────────────────

def get_model_summary(results) -> dict:
    """
    Extracts key information from a fitted ARIMA model.

    Returns:
        dict with:
          - "order"       : tuple(p,d,q)
          - "aic"         : float — Akaike Information Criterion (lower = better model)
          - "bic"         : float — Bayesian Information Criterion  (lower = better)
          - "coefficients": dict  — the learned AR and MA weights
          - "num_observations": int
          - "full_summary": str   — the full statsmodels text summary
          - "explanation" : str   — what these numbers mean
    """

    # ── AIC (Akaike Information Criterion) ────────────────────────────────
    # Measures model quality.  Balances fit accuracy vs. complexity.
    # LOWER AIC = BETTER MODEL.
    # If you try ARIMA(1,1,1) AIC=3500 vs ARIMA(2,1,2) AIC=3480,
    # the second one is slightly better.
    aic = float(results.aic)

    # ── BIC (Bayesian Information Criterion) ──────────────────────────────
    # Similar to AIC but penalizes complexity MORE.
    # Useful when you want a simpler model that still predicts well.
    bic = float(results.bic)

    # ── Coefficients ──────────────────────────────────────────────────────
    # These are the "weights" the model learned.
    # AR coefficient (ar.L1): how much yesterday's price affects today's
    # MA coefficient (ma.L1): how much yesterday's forecast error affects today's
    coefficients = {str(k): round(float(v), 6) for k, v in results.params.items()}

    # ── Full text summary ─────────────────────────────────────────────────
    full_summary = results.summary().as_text()

    explanation = (
        f"📊 Model: ARIMA{results.model.order}\n"
        f"   AIC = {aic:.1f}  (lower = better fit vs. complexity tradeoff)\n"
        f"   BIC = {bic:.1f}  (similar to AIC, penalizes complexity more)\n"
        f"   Observations used: {results.nobs}\n"
        f"\n   Learned coefficients:\n"
    )
    for name, val in coefficients.items():
        explanation += f"     {name} = {val}\n"

    return {
        "order":            results.model.order,
        "aic":              round(aic, 2),
        "bic":              round(bic, 2),
        "coefficients":     coefficients,
        "num_observations": int(results.nobs),
        "full_summary":     full_summary,
        "explanation":      explanation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 7:  save_model  /  load_model
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  Persist the trained model so you don't have to re-train
#           every time the app restarts.
#
# WHY:  Training takes a few seconds.  Once trained, save to disk.
#       The frontend teammate can then load the pre-trained model
#       instantly and just call forecast().
# ──────────────────────────────────────────────────────────────────────────────

def save_model(results, path: str = None) -> str:
    """
    Saves the trained ARIMA model to a .pkl file.

    Parameters:
        results : fitted ARIMA model
        path    : file path (default: SAVED_MODEL_PATH from config)

    Returns:
        str — the absolute path where the model was saved
    """
    if path is None:
        path = SAVED_MODEL_PATH

    # Create the directory if it doesn't exist
    # Example: if path = "model/saved_arima.pkl", we need the "model/" folder
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    # joblib.dump serializes the Python object into a binary file.
    # Think of it as "freezing" the trained model into a file.
    joblib.dump(results, path)

    return os.path.abspath(path)


def load_model(path: str = None):
    """
    Loads a previously saved ARIMA model from disk.

    Parameters:
        path : file path (default: SAVED_MODEL_PATH)

    Returns:
        The fitted ARIMA model object (same as what fit_arima returns)

    Raises:
        FileNotFoundError if the model file doesn't exist
    """
    if path is None:
        path = SAVED_MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'.  "
            f"You need to train and save the model first.  "
            f"Run:  python model/sample_runner.py"
        )

    # joblib.load "unfreezes" the model back into a Python object
    return joblib.load(path)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTION 8:  forecast_vnindex  (THE MAIN PUBLIC API)
# ══════════════════════════════════════════════════════════════════════════════
# PURPOSE:  The ONE function your teammates need to call.
#           It wraps everything above into a single call.
#
# YOUR DB TEAMMATE:  provides a DataFrame with "Date" and "Close" columns
# YOUR FRONTEND TEAMMATE:  receives a dictionary with predictions, dates,
#                           confidence intervals, accuracy metrics, and
#                           model summary — everything needed to display.
# ──────────────────────────────────────────────────────────────────────────────

def forecast_vnindex(
    df: pd.DataFrame,
    steps: int = None,
    order: tuple = None,
    save: bool = True,
) -> dict:
    """
    ★ MAIN API FUNCTION ★

    End-to-end: takes raw data, cleans it, fits ARIMA, evaluates,
    forecasts, and returns everything the frontend needs.

    Parameters:
        df    : pd.DataFrame with "Date" and "Close" columns (from DB)
        steps : how many days to forecast (default: FORECAST_STEPS = 7)
        order : ARIMA(p,d,q) order (default: DEFAULT_ORDER = (1,1,1))
        save  : whether to save the trained model to disk

    Returns:
        dict with:
          - "forecast"        : dict with dates, predictions, lower_ci, upper_ci
          - "evaluation"      : dict with rmse, mae, mape, actual vs predicted
          - "stationarity"    : dict with ADF test results
          - "model_summary"   : dict with AIC, BIC, coefficients
          - "last_known_price": float — the most recent closing price
          - "data_points"     : int   — how many data points were used
          - "order_used"      : tuple — the (p,d,q) that was used
    """

    if steps is None:
        steps = FORECAST_STEPS
    if order is None:
        order = DEFAULT_ORDER

    # Step 1: Clean the data
    series = prepare_data(df)

    # Step 2: Check stationarity (informational — ARIMA handles it via 'd')
    stationarity = check_stationarity(series)

    # Step 3: Evaluate accuracy (train/test split)
    evaluation = evaluate_model(series, order=order)

    # Step 4: Fit on ALL data (for the best possible forecast)
    results = fit_arima(series, order=order)

    # Step 5: Generate future predictions
    forecast_result = forecast(results, steps=steps)

    # Step 6: Get model summary
    summary = get_model_summary(results)

    # Step 7: Save model to disk (optional)
    saved_path = None
    if save:
        saved_path = save_model(results)

    # Step 8: Package everything
    return {
        "forecast":          forecast_result,
        "evaluation":        evaluation,
        "stationarity":      stationarity,
        "model_summary":     summary,
        "last_known_price":  round(float(series.iloc[-1]), 2),
        "last_known_date":   series.index[-1].strftime("%Y-%m-%d"),
        "data_points":       len(series),
        "order_used":        order,
        "saved_model_path":  saved_path,
    }
