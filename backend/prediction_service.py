from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from model import forecast_vnindex
from model.arima_config import DEFAULT_ORDER
from model.arima_model import fit_arima, prepare_data

# A compact candidate set for Auto-ARIMA-like order search by AIC.
CANDIDATE_ORDERS: Sequence[Tuple[int, int, int]] = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
    (3, 1, 1),
    (1, 0, 1),
    (2, 0, 2),
)

SUPPORTED_MODEL_TYPES = ("arima", "arimax")


def build_candidate_orders(use_log_returns: bool) -> List[Tuple[int, int, int]]:
    """Build candidate orders; raw-price mode excludes d=0 to avoid level disconnect."""
    if use_log_returns:
        return list(CANDIDATE_ORDERS)

    constrained = [order for order in CANDIDATE_ORDERS if int(order[1]) >= 1]
    return constrained or list(CANDIDATE_ORDERS)


def trend_label(current_price: float, predicted_price: float, neutral_threshold_pct: float = 0.4) -> str:
    """Classify trend as Upward/Downward/Sideways using percent change."""
    if current_price == 0:
        return "Sideways"

    pct_change = ((predicted_price - current_price) / current_price) * 100.0
    if pct_change > neutral_threshold_pct:
        return "Upward"
    if pct_change < -neutral_threshold_pct:
        return "Downward"
    return "Sideways"


def select_best_order(df: pd.DataFrame, candidate_orders: Optional[Iterable[Tuple[int, int, int]]] = None) -> Tuple[int, int, int]:
    """Pick ARIMA order with lowest AIC over a candidate grid."""
    series = prepare_data(df)
    orders = list(candidate_orders or CANDIDATE_ORDERS)

    best_order = DEFAULT_ORDER
    best_aic = float("inf")

    for order in orders:
        try:
            fitted = fit_arima(series, order=order)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
        except Exception:
            continue

    return best_order


def get_order_diagnostics(
    df: pd.DataFrame,
    model_type: str = "arima",
    use_sentiment: bool = False,
    use_volume: bool = False,
    candidate_orders: Optional[Iterable[Tuple[int, int, int]]] = None,
) -> pd.DataFrame:
    """Return per-order AIC/BIC diagnostics used by auto-selection."""
    mode = (model_type or "arima").lower().strip()
    orders = list(candidate_orders or CANDIDATE_ORDERS)
    rows: List[Dict[str, object]] = []

    if mode == "arimax":
        y, exog = _prepare_arimax_data(df, use_sentiment=use_sentiment, use_volume=use_volume)
        for order in orders:
            try:
                fitted = SARIMAX(
                    y,
                    exog=exog,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                rows.append(
                    {
                        "order": str(order),
                        "p": int(order[0]),
                        "d": int(order[1]),
                        "q": int(order[2]),
                        "aic": float(fitted.aic),
                        "bic": float(fitted.bic),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "order": str(order),
                        "p": int(order[0]),
                        "d": int(order[1]),
                        "q": int(order[2]),
                        "aic": np.nan,
                        "bic": np.nan,
                        "status": f"failed: {type(exc).__name__}",
                    }
                )
    else:
        series = prepare_data(df)
        for order in orders:
            try:
                fitted = fit_arima(series, order=order)
                rows.append(
                    {
                        "order": str(order),
                        "p": int(order[0]),
                        "d": int(order[1]),
                        "q": int(order[2]),
                        "aic": float(fitted.aic),
                        "bic": float(fitted.bic),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "order": str(order),
                        "p": int(order[0]),
                        "d": int(order[1]),
                        "q": int(order[2]),
                        "aic": np.nan,
                        "bic": np.nan,
                        "status": f"failed: {type(exc).__name__}",
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["order", "p", "d", "q", "aic", "bic", "status"])

    diagnostics = pd.DataFrame(rows)
    diagnostics["aic_rank"] = diagnostics["aic"].rank(method="dense", na_option="bottom")
    diagnostics = diagnostics.sort_values(["aic_rank", "bic"], na_position="last").reset_index(drop=True)
    return diagnostics


def _prepare_arimax_data(df: pd.DataFrame, use_sentiment: bool = False,
    use_volume: bool = False) -> tuple[pd.Series, pd.DataFrame]:
    base = df.copy()
    base["Date"] = pd.to_datetime(base["Date"])
    base["Close"] = pd.to_numeric(base["Close"], errors="coerce")
    base = base.dropna(subset=["Date", "Close"]).sort_values("Date").set_index("Date")

    # Align to business-day frequency so horizon indexing is consistent.
    close = base["Close"].asfreq("B").ffill()

    returns = close.pct_change().fillna(0.0)
    exog_dict = {
        "ret_1": returns,
        "ma_5": close.rolling(5, min_periods=1).mean(),
        "vol_5": returns.rolling(5, min_periods=1).std().fillna(0.0),
        "mom_3": close.diff(3).fillna(0.0),
    }


    if use_sentiment:
        # Simulate a market sentiment score [-1 to 1]
        np.random.seed(42)
        exog_dict["sentiment"] = np.clip(returns * 10.0 + np.random.normal(0, 0.5, len(returns)), -1, 1)

    if use_volume and "Volume" in base.columns:
        vol = pd.to_numeric(base["Volume"], errors="coerce").fillna(0)
        # Add 1 and log-transform to handle large numbers and 0s smoothly
        exog_dict["volume"] = np.log1p(vol).asfreq("B").ffill()


    exog = pd.DataFrame(exog_dict, index=close.index)
    close.name = "Close"
    return close, exog


def _build_future_exog(last_close: float, last_vol: float, steps: int, use_sentiment: bool = False,
    use_volume: bool = False, last_real_vol: float = 0.0) -> pd.DataFrame:
    # For unknown future exogenous variables, use conservative persistence assumptions.
    fut_exog = {
        "ret_1": np.zeros(steps),
        "ma_5": np.full(steps, last_close),
        "vol_5": np.full(steps, last_vol),
        "mom_3": np.zeros(steps),
    }

    if use_sentiment:
        fut_exog["sentiment"] = np.zeros(steps)  # Neutral future sentiment
    if use_volume:
        fut_exog["volume"] = np.full(steps, last_real_vol)  # Forward-fill last known volume

    return pd.DataFrame(fut_exog)


def select_best_order_arimax(
    df: pd.DataFrame,
    candidate_orders: Optional[Iterable[Tuple[int, int, int]]] = None,
    use_sentiment: bool = False,
    use_volume: bool = False,
) -> Tuple[int, int, int]:
    y, exog = _prepare_arimax_data(df, use_sentiment=use_sentiment, use_volume=use_volume)
    orders = list(candidate_orders or CANDIDATE_ORDERS)

    best_order = DEFAULT_ORDER
    best_aic = float("inf")
    for order in orders:
        try:
            model = SARIMAX(
                y,
                exog=exog,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
        except Exception:
            continue
    return best_order


def _predict_series_arima(
    df: pd.DataFrame,
    valid_horizons: Sequence[int],
    auto_order: bool,
    manual_order: Tuple[int, int, int],
    candidate_orders: Optional[Iterable[Tuple[int, int, int]]] = None,
) -> Dict[str, object]:
    steps = max(valid_horizons)
    order = select_best_order(df, candidate_orders=candidate_orders) if auto_order else manual_order

    result = forecast_vnindex(
        df=df,
        steps=steps,
        order=order,
        save=False,
    )

    last_price = float(result["last_known_price"])
    forecast_data = result["forecast"]

    rows: List[Dict[str, object]] = []
    for horizon in valid_horizons:
        idx = horizon - 1
        predicted = float(forecast_data["predictions"][idx])
        lower_ci = float(forecast_data["lower_ci"][idx])
        upper_ci = float(forecast_data["upper_ci"][idx])
        change_pct = ((predicted - last_price) / last_price) * 100.0 if last_price else 0.0
        rows.append(
            {
                "horizon_days": horizon,
                "predicted_price": round(predicted, 2),
                "change_pct": round(change_pct, 2),
                "trend": trend_label(last_price, predicted),
                "lower_ci": round(lower_ci, 2),
                "upper_ci": round(upper_ci, 2),
            }
        )

    summary_df = pd.DataFrame(rows)
    payload = {
        "selected_order": order,
        "last_known_price": last_price,
        "last_known_date": result["last_known_date"],
        "data_points": result["data_points"],
        "forecast": forecast_data,
        "evaluation": result["evaluation"],
        "stationarity": result["stationarity"],
        "model_summary": result["model_summary"],
        "horizon_table": summary_df,
    }
    return payload


def _predict_series_arimax(
    df: pd.DataFrame,
    valid_horizons: Sequence[int],
    auto_order: bool,
    manual_order: Tuple[int, int, int],
    use_sentiment: bool = False,
    use_volume: bool = False,
    candidate_orders: Optional[Iterable[Tuple[int, int, int]]] = None,
) -> Dict[str, object]:
    steps = max(valid_horizons)
    order = (
        select_best_order_arimax(
            df,
            candidate_orders=candidate_orders,
            use_sentiment=use_sentiment, use_volume=use_volume,
        )
        if auto_order
        else manual_order
    )

    y, exog = _prepare_arimax_data(df, use_sentiment=use_sentiment, use_volume=use_volume)
    if len(y) < 60:
        raise ValueError("Not enough data points for ARIMAX. Need at least 60 rows.")

    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    last_close = float(y.iloc[-1])
    last_vol = float(exog["vol_5"].iloc[-1]) if "vol_5" in exog.columns else 0.0
    last_real_vol = float(exog["volume"].iloc[-1]) if "volume" in exog.columns else 0.0
    future_exog = _build_future_exog(last_close, last_vol, steps, use_sentiment=use_sentiment, use_volume=use_volume, last_real_vol=last_real_vol)

    fc = fitted.get_forecast(steps=steps, exog=future_exog)
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    future_dates = pd.bdate_range(start=y.index[-1] + pd.Timedelta(days=1), periods=steps)
    forecast_data = {
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "predictions": [round(float(v), 2) for v in pred.values],
        "lower_ci": [round(float(v), 2) for v in ci.iloc[:, 0].values],
        "upper_ci": [round(float(v), 2) for v in ci.iloc[:, 1].values],
    }

    split = int(len(y) * 0.8)
    train_y, test_y = y.iloc[:split], y.iloc[split:]
    train_exog, test_exog = exog.iloc[:split], exog.iloc[split:]
    eval_model = SARIMAX(
        train_y,
        exog=train_exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    eval_fit = eval_model.fit(disp=False)
    eval_pred = eval_fit.forecast(steps=len(test_y), exog=test_exog)

    mae = float(mean_absolute_error(test_y.values, eval_pred.values))
    rmse = float(np.sqrt(mean_squared_error(test_y.values, eval_pred.values)))
    mape = float(np.mean(np.abs((test_y.values - eval_pred.values) / test_y.values)) * 100.0)

    rows: List[Dict[str, object]] = []
    for horizon in valid_horizons:
        idx = horizon - 1
        predicted = float(forecast_data["predictions"][idx])
        lower_ci = float(forecast_data["lower_ci"][idx])
        upper_ci = float(forecast_data["upper_ci"][idx])
        change_pct = ((predicted - last_close) / last_close) * 100.0 if last_close else 0.0
        rows.append(
            {
                "horizon_days": horizon,
                "predicted_price": round(predicted, 2),
                "change_pct": round(change_pct, 2),
                "trend": trend_label(last_close, predicted),
                "lower_ci": round(lower_ci, 2),
                "upper_ci": round(upper_ci, 2),
            }
        )

    summary_df = pd.DataFrame(rows)
    return {
        "selected_order": order,
        "last_known_price": round(last_close, 2),
        "last_known_date": y.index[-1].strftime("%Y-%m-%d"),
        "data_points": len(y),
        "forecast": forecast_data,
        "evaluation": {
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "train_size": len(train_y),
            "test_size": len(test_y),
            "actual": [round(float(v), 2) for v in test_y.values],
            "predicted": [round(float(v), 2) for v in eval_pred.values],
            "interpretation": (
                f"Model ARIMAX{order} — Evaluation on {len(test_y)} test days: "
                f"RMSE={rmse:.2f}, MAE={mae:.2f}, MAPE={mape:.2f}%"
            ),
        },
        "stationarity": {
            "is_stationary": None,
            "adf_statistic": None,
            "p_value": None,
            "critical_values": {},
            "interpretation": "ARIMAX mode uses exogenous regressors (ret_1, ma_5, vol_5, mom_3" + (", sentiment" if use_sentiment else "") + (", volume" if use_volume else "") + ").",
        },
        "model_summary": {
            "order": order,
            "aic": round(float(fitted.aic), 2),
            "bic": round(float(fitted.bic), 2),
            "coefficients": {k: round(float(v), 6) for k, v in fitted.params.items()},
            "num_observations": int(fitted.nobs),
            "full_summary": fitted.summary().as_text(),
            "explanation": (
                f"ARIMAX{order} with exogenous features ret_1, ma_5, vol_5, mom_3" + (", sentiment" if use_sentiment else "") + (", volume" if use_volume else "") + ". "
                f"AIC={float(fitted.aic):.2f}, BIC={float(fitted.bic):.2f}."
            ),
        },
        "horizon_table": summary_df,
    }


def get_acf_pacf(df: pd.DataFrame, lags: int = 15) -> tuple[list[float], list[float]]:
    try:
        from model.arima_model import prepare_data
        series = prepare_data(df)
        diff_series = series.diff().dropna()
        a = acf(diff_series, nlags=lags).tolist()
        p = pacf(diff_series, nlags=lags).tolist()
        return a, p
    except Exception:
        return [], []


def predict_series(
    df: pd.DataFrame,
    horizons: Sequence[int],
    auto_order: bool = True,
    manual_order: Tuple[int, int, int] = DEFAULT_ORDER,
    model_type: str = "arima",
    use_log_returns: bool = False,
    use_sentiment: bool = False,
    use_volume: bool = False,
) -> Dict[str, object]:
    """Run forecasting pipeline and return horizon-specific summaries."""
    valid_horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not valid_horizons:
        raise ValueError("Please provide at least one positive forecast horizon.")

    mode = (model_type or "arima").lower().strip()
    if mode not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose one of {SUPPORTED_MODEL_TYPES}.")

    candidate_orders = build_candidate_orders(use_log_returns=use_log_returns)

    if use_log_returns:
        df = df.copy()
        df["Close"] = np.log(df["Close"])

    if mode == "arimax":
        payload = _predict_series_arimax(
            df=df,
            valid_horizons=valid_horizons,
            auto_order=auto_order,
            manual_order=manual_order,
            use_sentiment=use_sentiment, use_volume=use_volume,
            candidate_orders=candidate_orders,
        )
    else:
        payload = _predict_series_arima(
            df=df,
            valid_horizons=valid_horizons,
            auto_order=auto_order,
            manual_order=manual_order,
            candidate_orders=candidate_orders,
        )
        
    if use_log_returns:
        payload["last_known_price"] = float(np.exp(payload["last_known_price"]))
        payload["forecast"]["predictions"] = [float(np.exp(x)) for x in payload["forecast"]["predictions"]]
        payload["forecast"]["lower_ci"] = [float(np.exp(x)) for x in payload["forecast"]["lower_ci"]]
        payload["forecast"]["upper_ci"] = [float(np.exp(x)) for x in payload["forecast"]["upper_ci"]]
        
        for idx, row in payload["horizon_table"].iterrows():
            pred = float(np.exp(row["predicted_price"]))
            last_p = payload["last_known_price"]
            change_pct = ((pred - last_p) / last_p) * 100.0 if last_p else 0.0
            
            payload["horizon_table"].at[idx, "predicted_price"] = pred
            payload["horizon_table"].at[idx, "change_pct"] = change_pct
            payload["horizon_table"].at[idx, "trend"] = trend_label(last_p, pred)
            payload["horizon_table"].at[idx, "lower_ci"] = float(np.exp(row["lower_ci"]))
            payload["horizon_table"].at[idx, "upper_ci"] = float(np.exp(row["upper_ci"]))

        if "actual" in payload["evaluation"] and "predicted" in payload["evaluation"]:
            act = [float(np.exp(x)) for x in payload["evaluation"]["actual"]]
            prd = [float(np.exp(x)) for x in payload["evaluation"]["predicted"]]
            mae = mean_absolute_error(act, prd)
            rmse = np.sqrt(mean_squared_error(act, prd))
            mape = np.mean(np.abs((np.array(act) - np.array(prd)) / np.array(act))) * 100.0
            payload["evaluation"]["mae"] = round(float(mae), 2)
            payload["evaluation"]["rmse"] = round(float(rmse), 2)
            payload["evaluation"]["mape"] = round(float(mape), 2)
            payload["evaluation"]["actual"] = act
            payload["evaluation"]["predicted"] = prd

    if auto_order:
        order_diag_df = get_order_diagnostics(
            df=df,
            model_type=mode,
            use_sentiment=use_sentiment, use_volume=use_volume,
            candidate_orders=candidate_orders,
        )
        payload["order_diagnostics"] = order_diag_df.to_dict(orient="records")
        ok_orders = order_diag_df[order_diag_df["status"] == "ok"].sort_values("aic")
        if not ok_orders.empty:
            best = ok_orders.iloc[0]
            runner_up = ok_orders.iloc[1] if len(ok_orders) > 1 else None
            if runner_up is not None:
                aic_gap = float(runner_up["aic"] - best["aic"])
                payload["order_selection_reason"] = (
                    f"Selected {best['order']} because it has the lowest AIC ({best['aic']:.2f}). "
                    f"Next best is {runner_up['order']} at AIC {runner_up['aic']:.2f} (gap {aic_gap:.2f})."
                )
            else:
                payload["order_selection_reason"] = (
                    f"Selected {best['order']} because it is the only successful candidate with AIC {best['aic']:.2f}."
                )

            if not use_log_returns:
                payload["order_selection_reason"] += (
                    " Raw-price mode excludes d=0 orders to avoid mean-reversion jumps that may disconnect "
                    "forecasts from the latest observed level."
                )
        else:
            payload["order_selection_reason"] = "Auto-selection fallback: no candidate successfully converged."
    else:
        payload["order_diagnostics"] = []
        payload["order_selection_reason"] = (
            f"Manual order mode is active. User-selected order: {payload.get('selected_order', manual_order)}."
        )

    payload["model_type"] = mode
    return payload
