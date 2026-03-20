# Stock Prediction Dashboard (PostgreSQL + ARIMAX)

This project builds an offline-ready **Streamlit dashboard** for stock forecasting using the SQL database from main branch (`stock.sql` schema style):
- metadata table: `public.company_info(symbol, icb_name2, ...)`
- per-symbol history tables: `symbol(time, close, ...)`

## Features
- Forecast modes: **Company** and **Sector Group** (sector mean close)
- Horizon output: **1 day / 3 days / 7 days**
- Direction labels: **⬆️ Upward / ⬇️ Downward / ➡️ Sideways**
- **ARIMAX** model via `statsmodels.SARIMAX`
- **Auto ARIMA order search** (AIC-based grid for `(p,d,q)`)
- Forecast chart with 95% confidence band
- Beginner explanation + debug panel + file logging (`app_debug.log`)
- Deterministic demo fallback if PostgreSQL is unavailable

## Quick start (offline)
1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure DB environment variables:
   - `DB_USER`
   - `DB_PASSWORD`
   - `DB_HOST`
   - `DB_PORT`
   - `DB_NAME`
3. Run app:
   ```bash
   streamlit run app.py
   ```

## Debug notes for beginners
- If DB connection fails, the app switches to demo data and still runs end-to-end.
- Open **Debug panel** in the app to see selected ARIMAX order and AIC/BIC.
- Check `app_debug.log` for backend diagnostics.
