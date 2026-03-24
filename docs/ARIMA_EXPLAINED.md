# 📘 ARIMA Explained — For Presentation & Judges

## 1. What is Time Series Data?

**Definition:** Data collected over time, in order. Each data point has a timestamp.

**VN-Index example:**
```
Date         Close
2024-01-02   1130.5
2024-01-03   1135.2    ← each row is one trading day
2024-01-04   1128.7
...
```

Stock prices are a **classic** time series because:
- They are recorded **daily** (every trading day)
- **Order matters** — Tuesday's price depends on Monday's
- There are **patterns** (trends, cycles) we can learn from

---

## 2. What is ARIMA?

ARIMA = **A**uto**R**egressive **I**ntegrated **M**oving **A**verage

It has **3 components**, each named after a letter:

### AR (AutoRegressive) — the "p" parameter
> "Today's price is partly based on **yesterday's price**."

Think of it like momentum:
- If the market went up yesterday, it's likely to go up a little today too
- `p = 1` means: look back 1 day
- `p = 2` means: look back 2 days
- Higher p = the model "remembers" further back

**Real-world analogy:** Like predicting weather — if it was hot yesterday, it's probably hot today too.

### I (Integrated) — the "d" parameter
> "Instead of predicting the **price**, predict the **change** in price."

Stock prices go up over years (trend). ARIMA can't work with trends, so we *remove* the trend by looking at **differences**:

```
Original:    1130, 1135, 1128, 1140
Differenced: +5, -7, +12              ← these have NO trend
```

- `d = 0` means: data has no trend, use as-is
- `d = 1` means: take one level of differencing (usually enough)
- `d = 2` means: difference twice (rarely needed)

**Real-world analogy:** Instead of saying "I weigh 70kg", you say "I gained 0.5kg" — the change is more predictable than the absolute value.

### MA (Moving Average) — the "q" parameter
> "Correct today's prediction based on **yesterday's mistake**."

If the model predicted 1130 but the actual was 1135 (error = +5), the MA term says: "I was wrong by +5, let me adjust today's prediction."

- `q = 1` means: learn from 1 past error
- Higher q = learn from more past errors

**Real-world analogy:** A student who checks their test answers — if they see they made a mistake last time, they adjust.

### Putting it together: ARIMA(1, 1, 1)
```
    Today's change ≈ (coefficient × yesterday's change)      ← AR
                    + (coefficient × yesterday's error)       ← MA
                    + noise

    Today's price  = yesterday's price + today's change       ← Integration (undo the differencing)
```

---

## 3. Why ARIMA is Best for This Data

### What type of data suits ARIMA?
| Characteristic | VN-Index | Suitable? |
|---|---|---|
| **Univariate** (single value over time) | ✅ Just closing prices | ✅ Perfect |
| **Sequential** (order matters) | ✅ Daily trading data | ✅ Perfect |
| **Moderate length** (100–2000 points) | ✅ ~500 trading days | ✅ Perfect |
| **Short-term forecast** (days, not years) | ✅ We predict 7 days | ✅ Perfect |
| **Reasonably smooth** (not random noise) | ✅ Stock indices are smooth | ✅ Good |

### ARIMA is BEST for:
- **Stock indices** (VN-Index, S&P 500, NIKKEI) — smooth, trending, daily data
- **Economic indicators** (GDP, inflation) — quarterly or yearly, slow-moving
- **Sales forecasting** — monthly revenue, seasonal patterns
- **Temperature data** — daily averages with yearly cycles

### ARIMA is NOT ideal for:
- **High-frequency trading** (millisecond data) → too noisy
- **Multi-variable problems** ("predict price using 50 features") → use ML instead
- **Very long horizons** (predict 2 years ahead) → forecast degrades quickly
- **Highly nonlinear data** (image recognition, NLP) → use deep learning

---

## 4. ARIMA vs Other Models

| Feature | ARIMA | XGBoost | LSTM (Deep Learning) | Prophet (Facebook) |
|---|---|---|---|---|
| **Type** | Statistical | Machine Learning | Neural Network | Decomposition |
| **Best for** | Univariate time series | Multi-feature tabular | Long sequences | Seasonal + holidays |
| **Interpretable?** | ✅ Very (you can read coefficients) | ⚠️ Somewhat | ❌ Black box | ✅ Yes |
| **Data needed** | 100+ points | 1000+ points | 5000+ points | 100+ points |
| **Training speed** | ⚡ Seconds | ⚡ Seconds | 🐢 Minutes-hours | ⚡ Seconds |
| **Handles trends?** | ✅ Via differencing (d) | ❌ Needs feature engineering | ✅ Automatically | ✅ Automatically |
| **Confidence intervals?** | ✅ Built-in | ❌ Not natively | ❌ Not natively | ✅ Built-in |
| **Setup complexity** | 🟢 Simple | 🟡 Medium | 🔴 Complex | 🟢 Simple |

### Why we chose ARIMA over others:

1. **Interpretability** — We can explain EVERY number to the judges. "The AR coefficient is 0.15, meaning 15% of today's change comes from yesterday's pattern." XGBoost and LSTM are harder to explain.

2. **Confidence intervals** — ARIMA gives us "the price will be between 1210 and 1260 with 95% confidence." This shows the judges we understand uncertainty. XGBoost doesn't do this natively.

3. **Perfect fit for VN-Index** — We have one variable (closing price), ~500 data points, and want a 7-day forecast. This is EXACTLY what ARIMA was designed for.

4. **Low data requirements** — We only have ~2 years of daily VN-Index data (~500 points). LSTM would need 5000+ to work well.

5. **Fast training** — Model trains in < 2 seconds. Great for live demos to judges.

---

## 5. How to Read the Model Output

When you run the model, it produces a summary. Here's how to read it:

### AIC and BIC (Information Criteria)
```
AIC = 3456.7
BIC = 3470.2
```
- **Lower = better model**
- AIC: "How well does the model fit vs how complex is it?"
- BIC: Same as AIC but penalizes complexity more
- Use these to compare different (p,d,q) settings — pick the one with lowest AIC

### Coefficients
```
ar.L1 = 0.15     ← AR coefficient (how much yesterday matters)
ma.L1 = -0.42    ← MA coefficient (how much yesterday's error matters)
sigma2 = 245.3   ← variance of the noise (lower = more predictable data)
```

**How to explain to judges:**
> "The AR coefficient of 0.15 means that 15% of today's price change is explained by yesterday's price change. The MA coefficient of -0.42 means the model corrects almost half of yesterday's forecast error."

### P-values of coefficients
```
ar.L1:  P>|z| = 0.002   ← GOOD (< 0.05, coefficient is statistically significant)
ma.L1:  P>|z| = 0.001   ← GOOD
```
- If P > 0.05, that term might not be helping — consider removing it

---

## 6. Key Vocabulary for Judges

| Term | Simple Meaning | Example |
|---|---|---|
| **Stationarity** | Data that doesn't trend up/down over time | Daily returns (% change) |
| **Differencing** | Subtracting yesterday from today | 1135 - 1130 = +5 |
| **ADF test** | Statistical test for stationarity | p < 0.05 = stationary |
| **AIC** | Model quality score (lower = better) | AIC = 3456 |
| **RMSE** | Average error in index points | RMSE = 15 points |
| **MAPE** | Average error as percentage | MAPE = 1.2% |
| **Confidence interval** | Range where the true value likely falls | [1210, 1260] at 95% |
| **Forecast horizon** | How far ahead we predict | 7 business days |
| **Overfitting** | Model memorizes training data, fails on new data | Train R²=0.99, Test R²=0.3 |

---

## 7. Presentation Script (What to Say to Judges)

> "We chose ARIMA for VN-Index prediction because it is the gold standard for univariate time series forecasting.
>
> Our data is daily closing prices over 2 years — about 500 data points. ARIMA is specifically designed for this type of data, unlike deep learning models that need thousands of points.
>
> The key advantage is **interpretability**. We can tell you exactly what the model learned: the AR coefficient tells us how much yesterday's pattern carries forward, and the MA coefficient shows how the model corrects its own mistakes.
>
> Plus, ARIMA gives us **confidence intervals** — we don't just say 'the price will be 1250,' we say 'we're 95% confident it'll be between 1230 and 1270.' This shows we understand uncertainty in financial data.
>
> Our model achieved a MAPE of approximately X%, meaning our average error is only X% of the actual price. For a 7-day forecast horizon, this is competitive with more complex models."
