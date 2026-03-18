"""
model/__init__.py  —  Public API
=================================
This file makes the 'model' folder into a Python package and
exposes the main function your teammates need.

USAGE FOR TEAMMATES:
    from model import forecast_vnindex
    result = forecast_vnindex(dataframe_from_db, steps=7)
"""

# Import the main API function so teammates can do:
#   from model import forecast_vnindex
# instead of:
#   from model.arima_model import forecast_vnindex
from model.arima_model import forecast_vnindex

# Also expose individual utilities for advanced use
from model.arima_model import (
    prepare_data,
    check_stationarity,
    fit_arima,
    forecast,
    evaluate_model,
    get_model_summary,
    save_model,
    load_model,
)

# Configuration constants
from model.arima_config import (
    DEFAULT_ORDER,
    FORECAST_STEPS,
    DATE_COL,
    CLOSE_COL,
)

__all__ = [
    "forecast_vnindex",
    "prepare_data",
    "check_stationarity",
    "fit_arima",
    "forecast",
    "evaluate_model",
    "get_model_summary",
    "save_model",
    "load_model",
    "DEFAULT_ORDER",
    "FORECAST_STEPS",
    "DATE_COL",
    "CLOSE_COL",
]
