# 🛠️ Debug Guide — Troubleshooting & Common Errors

> When things go wrong during integration or demo, check here first.

---

## Error 1: `ValueError: Column 'Date' not found`

**When:** You call `forecast_vnindex(df)` but the DataFrame doesn't have a "Date" column.

**Cause:** The DB teammate used a different column name like "date", "Ngay", or "timestamp".

**Fix:**
```python
# Rename the column BEFORE passing to the model
df = df.rename(columns={"your_date_column": "Date"})
```

**Or** change `DATE_COL` in `model/arima_config.py` to match the DB column name.

---

## Error 2: `ValueError: Column 'Close' not found`

Same issue but for the price column.

**Fix:**
```python
df = df.rename(columns={"Gia_dong_cua": "Close"})
```

---

## Error 3: `ValueError: Not enough data: got X rows, need at least 60`

**Cause:** The DB returned too few rows.

**Fix:**
1. Ask DB teammate to provide more historical data (at least 3 months)
2. Or temporarily lower `MIN_DATA_POINTS` in `arima_config.py` (not recommended for production)

---

## Error 4: `LinAlgError` or `ValueError: The computed initial AR coefficients are not stationary`

**When:** The ARIMA model fails to converge with the given (p,d,q).

**Cause:** The chosen parameters don't work well with this data.

**Fix — try different parameters:**
```python
# Try these common orders:
orders_to_try = [(1,1,1), (2,1,1), (1,1,2), (0,1,1), (1,2,1)]

for order in orders_to_try:
    try:
        result = forecast_vnindex(df, order=order)
        print(f"✅ ARIMA{order} worked!  MAPE = {result['evaluation']['mape']}%")
    except Exception as e:
        print(f"❌ ARIMA{order} failed: {e}")
```

---

## Error 5: Forecast is a FLAT line

**What it looks like:** All predicted values are nearly the same number.

**Cause:** Usually means `d=0` and the model is just predicting the mean.

**Fix:**
1. Set `d=1` (enable differencing)
2. Make sure you have enough data (200+ points ideal)
3. Check that the Close prices are NOT all the same value

---

## Error 6: `ModuleNotFoundError: No module named 'statsmodels'`

**Cause:** The conda environment isn't set up or not activated.

**Fix:**
```bash
conda activate adatralig
pip install statsmodels
```

Or recreate the environment:
```bash
conda env create -f environment.yml
conda activate adatralig
```

---

## Error 7: Yahoo Finance returns empty data

**When:** `sample_runner.py` shows "Failed to download data."

**Cause:** Internet issue, or Yahoo Finance is blocking requests, or the ticker symbol changed.

**Fix:**
1. Check your internet connection
2. Try manually: open `https://finance.yahoo.com/quote/%5EVNINDEX/` in your browser
3. If Yahoo doesn't work, create a CSV manually:
```python
import pandas as pd
df = pd.read_csv("your_vnindex_data.csv")  # Must have Date and Close columns
```

---

## Error 8: `FileNotFoundError: Model file not found`

**When:** Trying to load a model that hasn't been trained yet.

**Fix:**
```bash
# Train and save the model first
python model/sample_runner.py
# Then load it
```

---

## How to Verify the Model is Working Correctly

### Quick sanity check:
```python
import sys
sys.path.insert(0, ".")

from model import forecast_vnindex
import pandas as pd

# Create fake data that trends upward
df = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=200, freq="B"),
    "Close": [1200 + i * 0.5 for i in range(200)],
})

result = forecast_vnindex(df, steps=5, save=False)

# The forecast should CONTINUE the upward trend
last_price = result["last_known_price"]
first_prediction = result["forecast"]["predictions"][0]

print(f"Last known: {last_price}")
print(f"First prediction: {first_prediction}")

# Prediction should be close to (slightly above) last price
assert abs(first_prediction - last_price) < last_price * 0.05, \
    "Prediction is too far from last price — something is wrong!"

print("✅ Sanity check passed!")
```

### Full test checklist:
- [ ] `python model/sample_runner.py` runs without errors
- [ ] `python integration_example.py` runs without errors
- [ ] Forecast values are in realistic VN-Index range (900–1600)
- [ ] MAPE is below 5%
- [ ] Model file is saved at `model/saved_arima.pkl`
- [ ] `load_model()` returns a usable model object

---

## How to Compare Different ARIMA Parameters

```python
import sys
sys.path.insert(0, ".")

from model.arima_model import prepare_data, evaluate_model
import pandas as pd

# Load your data
df = pd.read_csv("data/your_data.csv")  # or get from DB
from model.arima_model import prepare_data
series = prepare_data(df)

# Compare orders
orders = [(1,1,0), (1,1,1), (2,1,1), (1,1,2), (2,1,2)]

print(f"{'Order':<12} {'RMSE':>8} {'MAE':>8} {'MAPE':>8}")
print("-" * 40)

for order in orders:
    try:
        ev = evaluate_model(series, order=order)
        print(f"{str(order):<12} {ev['rmse']:>8.2f} {ev['mae']:>8.2f} {ev['mape']:>7.2f}%")
    except:
        print(f"{str(order):<12}    FAILED")
```

Pick the order with the **lowest MAPE**.

---

## Emergency: Nothing Works Before the Demo

If EVERYTHING is broken 10 minutes before the demo:

```python
# Minimal working example — paste this into a Python file and run it
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Fake but realistic data
dates = pd.date_range("2024-01-01", periods=300, freq="B")
prices = 1200 + np.cumsum(np.random.normal(0.5, 10, 300))
df = pd.DataFrame({"Date": dates, "Close": prices})

series = df.set_index("Date")["Close"]
model = ARIMA(series, order=(1,1,1))
results = model.fit()
forecast = results.forecast(steps=7)

print("7-Day Forecast:")
print(forecast)
```

This runs in under 2 seconds with zero dependencies on your teammates' code.
