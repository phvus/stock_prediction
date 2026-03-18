"""
integration_example.py  —  How Teammates Connect
===================================================
This file shows EXACTLY how the DB and Frontend teammates integrate
with the ARIMA model module.

It is NOT part of the final app — it's a REFERENCE for your team.

Run it to verify integration works:
    conda activate adatralig
    python integration_example.py
"""

import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
import json


def example_db_teammate():
    """
    ═══════════════════════════════════════════════════════════════
    EXAMPLE:  What the DATABASE teammate needs to provide
    ═══════════════════════════════════════════════════════════════

    Your DB teammate must provide a pandas DataFrame with exactly
    two columns:
        "Date"  (str or datetime)  — the trading date
        "Close" (float)            — VN-Index closing price

    HOW they get it (from their SQLite / PostgreSQL / API):

        conn = sqlite3.connect("vnindex.db")
        df = pd.read_sql("SELECT Date, Close FROM vnindex_history", conn)
        conn.close()

    Then they pass 'df' to the model:

        from model import forecast_vnindex
        result = forecast_vnindex(df, steps=7)
    """
    print("=" * 60)
    print("  📦 EXAMPLE: DB Teammate → Model")
    print("=" * 60)

    # Simulate what the DB teammate provides:
    # A simple DataFrame with Date and Close columns
    sample_data = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=200, freq="B"),
        "Close": [1200 + i * 0.5 + (i % 7) * 3 for i in range(200)],
    })

    print(f"  DB provides a DataFrame with {len(sample_data)} rows:")
    print(f"  Columns: {list(sample_data.columns)}")
    print(f"  Date range: {sample_data['Date'].min()} → {sample_data['Date'].max()}")
    print()

    # ── Call the model ────────────────────────────────────────────────────
    from model import forecast_vnindex

    result = forecast_vnindex(
        df=sample_data,    # ← DataFrame from DB teammate
        steps=7,           # ← 7-day forecast
        order=(1, 1, 1),   # ← ARIMA(1,1,1) parameters
        save=False,        # ← don't save during example
    )

    return result


def example_frontend_teammate(result: dict):
    """
    ═══════════════════════════════════════════════════════════════
    EXAMPLE:  What the FRONTEND teammate receives
    ═══════════════════════════════════════════════════════════════

    The frontend teammate receives a dictionary with this structure:

    result = {
        "forecast": {
            "dates":       ["2024-10-14", "2024-10-15", ...],
            "predictions": [1305.12, 1306.45, ...],
            "lower_ci":    [1280.5, 1278.2, ...],
            "upper_ci":    [1329.8, 1334.7, ...],
        },
        "evaluation": {
            "rmse":  12.5,
            "mae":   10.3,
            "mape":  0.85,
            "interpretation": "Model ARIMA(1,1,1) — RMSE=12.5...",
        },
        "stationarity": {
            "is_stationary": False,
            "p_value":       0.42,
            "interpretation": "NON-STATIONARY...",
        },
        "model_summary": {
            "aic": 3456.7,
            "bic": 3470.2,
            "coefficients": {"ar.L1": 0.123, "ma.L1": -0.456},
            "explanation": "...",
        },
        "last_known_price": 1300.25,
        "last_known_date":  "2024-10-11",
        "data_points":      200,
        "order_used":       (1, 1, 1),
    }

    The frontend teammate can use this to:
      - Plot predictions vs confidence intervals
      - Show accuracy metrics (RMSE, MAPE)
      - Display model details in an expander
    """
    print()
    print("=" * 60)
    print("  🖥️  EXAMPLE: Model → Frontend Teammate")
    print("=" * 60)

    # Forecast data for charts
    fc = result["forecast"]
    print(f"\n  📊 Forecast ({len(fc['dates'])} days):")
    for i in range(len(fc["dates"])):
        print(f"    {fc['dates'][i]}:  {fc['predictions'][i]:.2f}  "
              f"[{fc['lower_ci'][i]:.2f} — {fc['upper_ci'][i]:.2f}]")

    # Accuracy for display
    ev = result["evaluation"]
    print(f"\n  📈 Accuracy:")
    print(f"    RMSE = {ev['rmse']}")
    print(f"    MAE  = {ev['mae']}")
    print(f"    MAPE = {ev['mape']}%")

    # Context info
    print(f"\n  📋 Context:")
    print(f"    Last price: {result['last_known_price']} on {result['last_known_date']}")
    print(f"    Model used: ARIMA{result['order_used']}")
    print(f"    AIC: {result['model_summary']['aic']}")

    # Show that it's JSON-serializable (for API endpoints)
    print(f"\n  🔗 JSON-serializable? ", end="")
    try:
        serializable = {
            "forecast": result["forecast"],
            "evaluation": {k: v for k, v in result["evaluation"].items()
                          if k != "actual" and k != "predicted"},
            "last_known_price": result["last_known_price"],
            "order_used": result["order_used"],
        }
        json_str = json.dumps(serializable, indent=2)
        print("✅ Yes — can be sent as API response")
    except Exception as e:
        print(f"❌ {e}")


if __name__ == "__main__":
    print()
    result = example_db_teammate()
    example_frontend_teammate(result)
    print()
    print("=" * 60)
    print("  ✅ Integration verified!  Both sides work correctly.")
    print("=" * 60)
    print()
