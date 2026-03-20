import datetime as dt
import logging
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from statsmodels.tsa.statespace.sarimax import SARIMAX

LOGGER = logging.getLogger("stock_forecast")
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("app_debug.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

TREND_THRESHOLD = 0.2


def get_db_config() -> dict:
    return {
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", "postgres"),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "stock"),
    }


def safe_table_name(symbol: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", symbol):
        raise ValueError(f"Unsafe symbol/table name: {symbol}")
    return symbol.lower()


def get_connection():
    return psycopg2.connect(**get_db_config())


def load_companies() -> pd.DataFrame:
    conn = get_connection()
    try:
        query = """
        SELECT symbol, COALESCE(icb_name2, 'Unknown') AS sector
        FROM public.company_info
        ORDER BY symbol
        """
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def load_symbol_history(symbol: str, years: int = 2) -> pd.DataFrame:
    table = safe_table_name(symbol)
    start_date = dt.date.today() - dt.timedelta(days=365 * years)
    conn = get_connection()
    try:
        query = sql.SQL(
            'SELECT time AS "Date", close AS "Close" FROM {} WHERE time >= %s ORDER BY time'
        ).format(sql.Identifier(table))
        with conn.cursor() as cur:
            cur.execute(query, (start_date,))
            rows = cur.fetchall()
        return pd.DataFrame(rows, columns=["Date", "Close"])
    finally:
        conn.close()


def load_sector_history(companies: pd.DataFrame, sector: str, years: int = 2) -> tuple[pd.DataFrame, list[str]]:
    symbols = companies.loc[companies["sector"] == sector, "symbol"].dropna().tolist()
    all_histories: list[pd.DataFrame] = []
    failed_symbols: list[str] = []
    for symbol in symbols:
        try:
            history = load_symbol_history(symbol, years=years)
            if not history.empty:
                history["Symbol"] = symbol
                all_histories.append(history)
        except Exception as exc:
            LOGGER.warning("Failed loading %s: %s", symbol, exc)
            failed_symbols.append(symbol)

    if not all_histories:
        return pd.DataFrame(columns=["Date", "Close"]), failed_symbols

    merged = pd.concat(all_histories, ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"])
    result = merged.groupby("Date", as_index=False)["Close"].mean().sort_values("Date")
    return result, failed_symbols


def build_exog_features(dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    idx = pd.to_datetime(pd.Series(dates))
    dow = idx.dt.dayofweek
    month = idx.dt.month
    n = np.arange(len(idx))
    return pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * dow / 7),
            "dow_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "trend": n,
        }
    )


def pick_auto_order(y: pd.Series, exog: pd.DataFrame, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> tuple[tuple[int, int, int], list[dict]]:
    best_order = None
    best_aic = float("inf")
    trials: list[dict] = []
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = SARIMAX(
                        y,
                        exog=exog,
                        order=(p, d, q),
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)
                    aic = float(fitted.aic)
                    trials.append({"order": (p, d, q), "aic": aic, "status": "ok"})
                    if np.isfinite(aic) and aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                except Exception as exc:
                    trials.append({"order": (p, d, q), "aic": None, "status": f"fail: {exc}"})

    if best_order is None:
        raise RuntimeError("Auto ARIMAX failed for all candidate orders.")
    return best_order, trials


def classify_trend(last_price: float, predicted_price: float) -> str:
    pct = ((predicted_price - last_price) / last_price) * 100
    if pct > TREND_THRESHOLD:
        return "⬆️ Upward"
    if pct < -TREND_THRESHOLD:
        return "⬇️ Downward"
    return "➡️ Sideways"


def forecast_arimax(
    df: pd.DataFrame,
    steps: int = 7,
    auto_order: bool = True,
    manual_order: tuple[int, int, int] = (1, 1, 1),
) -> dict:
    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Date", "Close"]).sort_values("Date")

    if len(data) < 40:
        raise ValueError("Not enough historical points; need at least 40 rows.")

    y = data["Close"].astype(float).reset_index(drop=True)
    train_exog = build_exog_features(data["Date"]).reset_index(drop=True)

    if auto_order:
        chosen_order, order_trials = pick_auto_order(y, train_exog)
    else:
        chosen_order = manual_order
        order_trials = [{"order": manual_order, "aic": None, "status": "manual"}]

    LOGGER.info("Training ARIMAX with order=%s rows=%d", chosen_order, len(data))
    model = SARIMAX(
        y,
        exog=train_exog,
        order=chosen_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    future_dates = pd.date_range(
        start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=steps,
        freq="B",
    )
    forecast_exog = build_exog_features(future_dates)
    fc = fitted.get_forecast(steps=steps, exog=forecast_exog)
    preds = fc.predicted_mean.astype(float).tolist()
    conf = fc.conf_int(alpha=0.05)

    return {
        "chosen_order": chosen_order,
        "last_known_price": float(y.iloc[-1]),
        "history": {
            "dates": data["Date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices": data["Close"].astype(float).round(2).tolist(),
        },
        "forecast": {
            "dates": future_dates.strftime("%Y-%m-%d").tolist(),
            "predictions": [round(v, 2) for v in preds],
            "lower_95": [round(float(v), 2) for v in conf.iloc[:, 0].tolist()],
            "upper_95": [round(float(v), 2) for v in conf.iloc[:, 1].tolist()],
        },
        "debug": {
            "num_history_rows": int(len(data)),
            "order_trials": order_trials,
            "aic": float(fitted.aic),
            "bic": float(fitted.bic),
            "log_likelihood": float(fitted.llf),
            "note": "Auto ARIMAX selects order with lowest AIC among tested combinations.",
        },
        "explanation": {
            "beginner": (
                "ARIMAX predicts future prices using past prices (ARIMA) plus extra signals (X). "
                "Here X uses calendar effects (weekday/month cycles and trend)."
            )
        },
    }


def summarize_horizons(result: dict, selected_horizons: list[int]) -> pd.DataFrame:
    last_price = float(result["last_known_price"])
    dates = result["forecast"]["dates"]
    preds = result["forecast"]["predictions"]
    rows = []
    for h in sorted(selected_horizons):
        idx = h - 1
        pred = float(preds[idx])
        change = ((pred - last_price) / last_price) * 100
        rows.append(
            {
                "Horizon": f"{h} day" if h == 1 else f"{h} days",
                "Forecast Date": dates[idx],
                "Predicted Close": round(pred, 2),
                "Change (%)": round(change, 2),
                "Trend": classify_trend(last_price, pred),
            }
        )
    return pd.DataFrame(rows)
