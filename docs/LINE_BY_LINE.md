# 🔍 Line-by-Line Code Explanation

> Every function in `model/arima_model.py` explained in plain English.
> Read this before your presentation so you can answer ANY judge question.

---

## Imports Section

```python
import warnings
warnings.filterwarnings("ignore")
```
**What:** Tells Python to suppress warning messages.
**Why:** ARIMA sometimes shows "convergence warnings" that are technical and not errors. They would clutter the output and confuse users. We silence them.
**If you remove it:** The output will have noisy yellow warnings — not harmful, just ugly.

```python
import numpy as np
```
**What:** NumPy = Numerical Python. The fundamental math library.
**Why:** We need it for `sqrt()` (square root), `mean()` (average), `abs()` (absolute value), and array operations. Every ML library depends on NumPy.

```python
import pandas as pd
```
**What:** Pandas = "Panel Data." Provides DataFrames — basically spreadsheets in Python.
**Why:** Our VN-Index data is a table with Date and Close columns. Pandas lets us filter, sort, clean, and manipulate this table easily.

```python
import joblib
```
**What:** A library for saving/loading Python objects to files.
**Why:** After training the ARIMA model (which takes a few seconds), we save it to disk so we don't have to retrain every time the app restarts. `joblib` is faster and smaller than Python's built-in `pickle` for numerical data.

```python
from statsmodels.tsa.arima.model import ARIMA
```
**What:** The actual ARIMA model implementation.
**Why:** `statsmodels` is the go-to Python library for statistical models. `tsa` = Time Series Analysis. This gives us the `ARIMA` class that does all the math.

```python
from statsmodels.tsa.stattools import adfuller
```
**What:** The Augmented Dickey-Fuller test.
**Why:** Before running ARIMA, we need to check if our data is "stationary" (no trend). This test gives us a p-value — if p < 0.05, the data is stationary.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
```
**What:** Standard metrics to measure prediction accuracy.
**Why:** We need to tell the judges HOW GOOD our model is. RMSE and MAE are the standard ways to do that.

---

## Function 1: `prepare_data(df)`

### Purpose
Takes the raw DataFrame from the DB teammate and makes it ready for ARIMA.

### Line by Line

```python
if DATE_COL not in df.columns:
    raise ValueError(...)
```
**What:** Checks that the DataFrame has a "Date" column.
**Why:** If the DB teammate sends data without a Date column, we want a CLEAR error message, not a mysterious crash 50 lines later. **Fail fast, fail loud.**

```python
data = df[[DATE_COL, CLOSE_COL]].copy()
```
**What:** Takes only the Date and Close columns, and makes a copy.
**Why:** The DB might send extra columns (Open, High, Low, Volume). We don't need them. `.copy()` prevents us from accidentally modifying the original DataFrame.

```python
data[DATE_COL] = pd.to_datetime(data[DATE_COL])
```
**What:** Converts date strings like `"2024-01-15"` into Python datetime objects.
**Why:** Pandas needs proper datetime objects to sort dates correctly and to set the time frequency (business days).

```python
data = data.sort_values(DATE_COL)
```
**What:** Sorts rows by date, oldest first.
**Why:** ARIMA expects chronological order. Database queries don't guarantee order, so we always sort.

```python
data = data.set_index(DATE_COL)
```
**What:** Makes the Date column the "index" (row label) instead of a regular column.
**Why:** ARIMA in `statsmodels` works best when the time column is the index. It also enables neat slicing like `series["2024-01":"2024-06"]`.

```python
data[CLOSE_COL] = pd.to_numeric(data[CLOSE_COL], errors="coerce")
```
**What:** Forces the Close column to be a number (float).
**Why:** Sometimes databases export prices as strings like `"1,234.50"`. `errors="coerce"` turns unparseable values into NaN instead of crashing.

```python
data = data.dropna()
```
**What:** Removes rows with missing values (NaN).
**Why:** ARIMA cannot handle gaps. Missing data usually comes from holidays when the market was closed.

```python
if len(data) < MIN_DATA_POINTS:
    raise ValueError(...)
```
**What:** Checks we have at least 60 data points.
**Why:** With only 10 points, ARIMA can't learn any patterns. 60 is about 3 months of daily data — the minimum for a meaningful model.

```python
series = data[CLOSE_COL].asfreq("B")
series = series.ffill()
```
**What:** Sets a "business day" frequency and fills any missing dates.
**Why:** ARIMA needs to know the data frequency. `"B"` = business days (Mon-Fri). `ffill()` = "forward fill" — copies the previous day's price for any gap (e.g., a random holiday not in the data).

---

## Function 2: `check_stationarity(series)`

```python
result = adfuller(series.dropna(), autolag="AIC")
```
**What:** Runs the Augmented Dickey-Fuller statistical test.
**Why:** Tests the **null hypothesis** "this data has a unit root (is NOT stationary)."
- If we can REJECT this hypothesis (p < 0.05), then the data IS stationary.
- `autolag="AIC"`: automatically picks the best number of lags to test using AIC criterion.

```python
adf_stat = result[0]   # Test statistic (more negative = more stationary)
p_value  = result[1]   # Probability data is NOT stationary
```

```python
is_stationary = p_value < ADF_SIGNIFICANCE  # ADF_SIGNIFICANCE = 0.05
```
**What:** If p-value < 5%, we're 95% confident the data is stationary.
**Why:** ARIMA assumes stationarity. If data isn't stationary, the `d` parameter in ARIMA will handle it by differencing.

---

## Function 3: `fit_arima(series, order)`

```python
model = ARIMA(series, order=order)
```
**What:** Creates the ARIMA model structure. Does NOT train it yet.
**Why:** Think of this as building the skeleton — we define "I want an ARIMA with these (p,d,q) settings on this data."

```python
results = model.fit()
```
**What:** Trains the model using Maximum Likelihood Estimation (MLE).
**Why:** This is where the math happens. The model tries millions of coefficient combinations and picks the ones that best explain the historical data. Like finding the "best fit line" in linear regression, but for time series.

---

## Function 4: `forecast(results, steps)`

```python
forecast_result = results.get_forecast(steps=steps)
```
**What:** Asks the trained model to predict `steps` days into the future.
**Why:** Unlike `.forecast()` which only gives values, `.get_forecast()` also gives confidence intervals.

```python
predicted_values = forecast_result.predicted_mean
```
**What:** Extracts the point predictions (e.g., [1235.2, 1237.8, ...]).

```python
alpha = 1 - CONFIDENCE_LEVEL  # 1 - 0.95 = 0.05
confidence_intervals = forecast_result.conf_int(alpha=alpha)
```
**What:** Gets the 95% confidence interval for each prediction.
**Why:** Instead of just saying "price will be 1235", we say "we're 95% sure it'll be between 1210 and 1260." This shows the judges we understand uncertainty.

---

## Function 5: `evaluate_model(series, order)`

```python
split_point = int(len(series) * (1 - TEST_SIZE_RATIO))
train = series[:split_point]
test  = series[split_point:]
```
**What:** Splits data into 80% training and 20% testing.
**Why:** We train on the first 80% and test on the last 20% to see how well the model predicts UNSEEN data. If we tested on training data, we'd be cheating (overfitting).

```python
rmse_val = float(np.sqrt(mean_squared_error(actual, predicted)))
```
**What:** RMSE = Root Mean Squared Error.
**Why:** Tells us "on average, our prediction is off by X index points." It penalizes big errors more than small ones because of the squaring.

```python
mape_val = float(np.mean(np.abs((actual - predicted) / actual)) * 100)
```
**What:** MAPE = Mean Absolute Percentage Error.
**Why:** The most intuitive metric. "Our predictions are off by 1.5% on average." Judges love this because it's easy to understand.

---

## Function 6: `get_model_summary(results)`

```python
aic = float(results.aic)
bic = float(results.bic)
```
**What:** AIC and BIC are "report cards" for the model.
**Why:** Lower = better. Used to compare different (p,d,q) settings. If ARIMA(1,1,1) has AIC=3500 and ARIMA(2,1,1) has AIC=3480, the second is slightly better.

```python
coefficients = {str(k): round(float(v), 6) for k, v in results.params.items()}
```
**What:** Extracts the learned weights (AR and MA coefficients).
**Why:** These are what the model "learned." You can explain: "The AR coefficient of 0.15 means 15% of today's change comes from yesterday's."

---

## Function 7: `save_model` / `load_model`

```python
joblib.dump(results, path)
```
**What:** Serializes the trained model to a `.pkl` file.
**Why:** Training takes seconds but we don't want to redo it every time. Save once, load instantly later.

```python
joblib.load(path)
```
**What:** Reads the `.pkl` file back into a Python object.
**Why:** Frontend teammate loads this to display predictions without re-training.

---

## Function 8: `forecast_vnindex(df, steps, order, save)`

This is the **master function** that chains everything together:

```python
series       = prepare_data(df)           # Step 1: Clean
stationarity = check_stationarity(series) # Step 2: Test
evaluation   = evaluate_model(series)     # Step 3: Evaluate (train/test)
results      = fit_arima(series)          # Step 4: Train on ALL data
forecast_result = forecast(results)       # Step 5: Predict future
summary      = get_model_summary(results) # Step 6: Get stats
save_model(results)                       # Step 7: Save to disk
```

**Why all data for final training?** We evaluated accuracy on 80/20 split (Step 3), but for the actual forecast we use ALL data (Step 4) because more data = better model. The evaluation already proved the model works.
