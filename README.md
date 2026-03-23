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
