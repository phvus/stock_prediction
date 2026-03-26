# Stock Prediction (Model Branch)

This branch combines:
- Main-branch database ingestion logic (PostgreSQL + VN100 import)
- Model-branch ARIMA forecasting logic
- Streamlit visualization for company and sector-group prediction

## Features

- PostgreSQL as primary data source
- Works with both database layouts:
  - Legacy layout: one table per symbol + `company_info`
  - Normalized layout: `stock_prices` + `company_info`
- Forecast horizons: 1 day, 3 days, 7 days, or custom day selection
- Trend labels: `Upward`, `Downward`, `Sideways`
- Sector grouping and batch sector prediction scan
- Auto-ARIMA order selection (AIC-based search over common orders)
- Optional Auto-GARCH volatility modeling (AIC/BIC model selection, Normal/Student-t)
- Phase 2 backtest now includes GARCH-aware diagnostics:
  - High-volatility hit rate (<1% error)
  - Volatility-error correlation

## Environment Variables

Create a `.env` file in project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock
DB_USER=postgres
DB_PASSWORD=your_password
```

## Install

```bash
pip install -r requirements.txt
```

## Prepare Database

1. Create schema manually (optional):

```sql
-- run stock.sql
```

2. Import VN100 data:

```bash
python Import_Data.py --symbols-file VN100.txt --legacy-tables
```

Optional custom dates:

```bash
python Import_Data.py --start 2024-01-01 --end 2026-03-01
```

## Run App

```bash
streamlit run app.py
```

In the app, you can:
- Select company symbol or sector group
- Select prediction days (1/3/7 or custom)
- Use auto ARIMA order search or manual `(p,d,q)`
- View chart, confidence interval, metrics, and trend direction
- Run sector group scan to compare sectors

## Phase 2 GARCH Validation (Historical Backtest)

When GARCH is enabled in Phase 1, the same volatility signal is now propagated into Phase 2 backtest rows.

Two additional validation results are computed for each horizon:

1. `High-Vol Hit Rate (<1% Err)`
  - Build the high-volatility bucket from the top 30% of `forecast_volatility` values in the selected horizon.
  - Compute the share of samples in that bucket where absolute percentage error is <= 1%.

2. `Volatility-Error Correlation`
  - Compute Pearson correlation between `forecast_volatility` and `abs_pct_error`.
  - Positive values suggest larger forecasted volatility tends to coincide with larger realized forecast error.

These diagnostics are shown in the Phase 2 panel as metric cards and an interactive chart overlaying forecast volatility against absolute percentage error.
