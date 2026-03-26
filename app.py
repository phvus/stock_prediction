from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import PostgresRepository, ApiRepository, predict_series, get_acf_pacf

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("Stock Prediction Dashboard")
st.caption(
    "Forecasting for company symbols and sector groups with Auto-ARIMA order selection (Supports PostgreSQL or VNStock API)."
)


@st.cache_resource
def get_postgres_repo() -> PostgresRepository:
    return PostgresRepository()


@st.cache_resource
def get_api_repo() -> ApiRepository:
    return ApiRepository()


@st.cache_data(ttl=600)
def load_companies(_repo, source: str) -> pd.DataFrame:
    return _repo.get_companies()


@st.cache_data(ttl=600)
def load_symbol_history(_repo, source: str, symbol: str) -> pd.DataFrame:
    return _repo.get_symbol_history(symbol)


@st.cache_data(ttl=600)
def load_sector_history(_repo, source: str, sector: str) -> pd.DataFrame:
    return _repo.get_sector_history(sector)





def make_order_diagnostic_chart(order_diag_df: pd.DataFrame, selected_order: tuple[int, int, int]) -> go.Figure:
    chart_df = order_diag_df[order_diag_df["status"] == "ok"].copy()
    if chart_df.empty:
        return go.Figure()

    chart_df = chart_df.sort_values("aic")
    selected_str = str(tuple(selected_order))
    colors = ["#d62728" if o == selected_str else "#1f77b4" for o in chart_df["order"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["order"],
            y=chart_df["aic"],
            name="AIC",
            marker_color=colors,
            text=[f"{v:.1f}" for v in chart_df["aic"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["order"],
            y=chart_df["bic"],
            mode="lines+markers",
            name="BIC",
            line=dict(color="#2ca02c", width=2),
        )
    )
    fig.update_layout(
        title="Auto-Order Candidate Test Results (Lower AIC/BIC Is Better)",
        xaxis_title="Tested (p,d,q)",
        yaxis_title="Information Criterion",
        template="plotly_white",
        height=420,
        hovermode="x unified",
    )
    return fig


def make_pdq_map_chart(order_diag_df: pd.DataFrame, selected_order: tuple[int, int, int]) -> go.Figure:
    chart_df = order_diag_df[order_diag_df["status"] == "ok"].copy()
    if chart_df.empty:
        return go.Figure()

    aic_min = chart_df["aic"].min()
    aic_max = chart_df["aic"].max()
    span = max(aic_max - aic_min, 1e-9)
    chart_df["quality"] = 1.0 - ((chart_df["aic"] - aic_min) / span)
    chart_df["size"] = 12 + (chart_df["quality"] * 22)

    fig = go.Figure()
    for d_value in sorted(chart_df["d"].unique().tolist()):
        d_df = chart_df[chart_df["d"] == d_value]
        fig.add_trace(
            go.Scatter(
                x=d_df["p"],
                y=d_df["q"],
                mode="markers+text",
                text=d_df["order"],
                textposition="top center",
                name=f"d={int(d_value)}",
                marker=dict(size=d_df["size"], opacity=0.8),
                customdata=d_df[["aic", "bic"]],
                hovertemplate="order=%{text}<br>p=%{x}, q=%{y}<br>AIC=%{customdata[0]:.2f}<br>BIC=%{customdata[1]:.2f}<extra></extra>",
            )
        )

    sel = chart_df[
        (chart_df["p"] == int(selected_order[0]))
        & (chart_df["d"] == int(selected_order[1]))
        & (chart_df["q"] == int(selected_order[2]))
    ]
    if not sel.empty:
        fig.add_trace(
            go.Scatter(
                x=sel["p"],
                y=sel["q"],
                mode="markers",
                name="Selected Order",
                marker=dict(size=24, symbol="star", color="#d62728", line=dict(width=1, color="#111")),
                hovertemplate="selected=%{text}<extra></extra>",
                text=sel["order"],
            )
        )

    fig.update_layout(
        title="PDQ Search Map (Auto-ARIMA/ARIMAX Tested Orders)",
        xaxis_title="p (AR lags)",
        yaxis_title="q (MA lags)",
        template="plotly_white",
        height=430,
    )
    return fig


def make_month_window_chart(history_df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> go.Figure:
    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")

    start_month = pd.Timestamp(start_date).to_period("M").to_timestamp()
    end_month = pd.Timestamp(end_date).to_period("M").to_timestamp("M")
    month_df = df[(df["Date"] >= start_month) & (df["Date"] <= end_month)].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=month_df["Date"],
            y=month_df["Close"],
            mode="lines+markers",
            name="Close",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_vrect(
        x0=pd.Timestamp(start_date),
        x1=pd.Timestamp(end_date),
        fillcolor="rgba(255, 127, 14, 0.18)",
        line_width=0,
        annotation_text="Selected validation range",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Full Month View For Selected Validation Timeframe",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=420,
        hovermode="x unified",
    )
    return fig


def compute_profit_projection(
    details_df: pd.DataFrame,
    selected_horizon: int,
    initial_capital: float,
) -> tuple[dict, pd.DataFrame]:
    horizon_df = (
        details_df[details_df["horizon_days"] == selected_horizon]
        .sort_values("cutoff_date")
        .copy()
    )
    if horizon_df.empty:
        return {}, pd.DataFrame()

    direction = np.where(horizon_df["predicted_price"] >= horizon_df["base_price"], 1.0, -1.0)
    realized_move = (horizon_df["actual_price"] - horizon_df["base_price"]) / horizon_df["base_price"]
    horizon_df["strategy_return_pct"] = direction * realized_move * 100.0
    horizon_df["equity"] = initial_capital * (1.0 + (horizon_df["strategy_return_pct"] / 100.0)).cumprod()

    avg_return_pct = float(horizon_df["strategy_return_pct"].mean())
    cumulative_return_pct = float((horizon_df["equity"].iloc[-1] / initial_capital - 1.0) * 100.0)

    years_observed = max((len(horizon_df) * selected_horizon) / 252.0, 1e-6)
    annualized_return = (horizon_df["equity"].iloc[-1] / initial_capital) ** (1.0 / years_observed) - 1.0
    annualized_return_pct = float(annualized_return * 100.0)
    projected_year_profit_value = float(initial_capital * annualized_return)

    return (
        {
            "trades": int(len(horizon_df)),
            "avg_return_pct": avg_return_pct,
            "cumulative_return_pct": cumulative_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "projected_year_profit_value": projected_year_profit_value,
        },
        horizon_df,
    )


def make_profit_curve_chart(profit_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if profit_df.empty:
        return fig

    fig.add_trace(
        go.Scatter(
            x=profit_df["target_date"],
            y=profit_df["equity"],
            mode="lines+markers",
            name="Strategy Equity",
            line=dict(color="#2ca02c", width=3),
        )
    )
    fig.update_layout(
        title="Strategy Equity Curve From Validation Trades",
        xaxis_title="Target Date",
        yaxis_title="Portfolio Value",
        template="plotly_white",
        height=420,
        hovermode="x unified",
    )
    return fig


def make_garch_forecast_overlay_chart(horizon_table: pd.DataFrame) -> go.Figure:
    df = horizon_table.copy().sort_values("horizon_days")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["horizon_days"],
            y=df["upper_ci"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["horizon_days"],
            y=df["lower_ci"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(width=0),
            name="Price CI",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["horizon_days"],
            y=df["predicted_price"],
            mode="lines+markers",
            name="Predicted Price",
            line=dict(color="#1f77b4", width=2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["horizon_days"],
            y=df["forecast_volatility"],
            name="Forecast Volatility",
            marker_color="#d62728",
            opacity=0.45,
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="ARIMAX Price Path + GARCH Volatility By Horizon",
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Horizon (days)",
        yaxis=dict(title="Predicted Price"),
        yaxis2=dict(title="Volatility", overlaying="y", side="right", showgrid=False),
        height=500,
    )
    return fig


def make_garch_volatility_term_structure_chart(horizon_table: pd.DataFrame) -> go.Figure:
    df = horizon_table.copy().sort_values("horizon_days")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["horizon_days"],
            y=df["forecast_volatility"],
            mode="lines+markers",
            name="Volatility",
            line=dict(color="#d62728", width=3),
        )
    )
    fig.update_layout(
        title="GARCH Volatility Term Structure",
        template="plotly_white",
        xaxis_title="Horizon (days)",
        yaxis_title="Forecast Volatility",
        height=420,
        hovermode="x unified",
    )
    return fig


def make_garch_search_heatmap(trials_df: pd.DataFrame, criterion_col: str) -> go.Figure:
    ok_df = trials_df[trials_df["status"] == "ok"].copy()
    fig = go.Figure()
    if ok_df.empty:
        return fig

    pivot = ok_df.pivot_table(index="p", columns="q", values=criterion_col, aggfunc="min")
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[int(v) for v in pivot.columns.tolist()],
            y=[int(v) for v in pivot.index.tolist()],
            colorscale="Viridis",
            colorbar=dict(title=criterion_col.upper()),
            hovertemplate="p=%{y}<br>q=%{x}<br>score=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Auto-GARCH Search Heatmap ({criterion_col.upper()})",
        template="plotly_white",
        xaxis_title="q",
        yaxis_title="p",
        height=420,
    )
    return fig


def make_smart_unified_forecast_chart(
    history_df: pd.DataFrame,
    prediction: dict,
    include_volume: bool = False,
) -> go.Figure:
    hist_df = history_df.copy()
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    hist_df["Close"] = pd.to_numeric(hist_df["Close"], errors="coerce")
    hist_df = hist_df.dropna(subset=["Date", "Close"]).sort_values("Date")

    # Focus on recent history so forecast structure remains readable.
    hist_tail = hist_df.tail(120)

    forecast = prediction.get("forecast", {})
    fc_dates = pd.to_datetime(pd.Series(forecast.get("dates", [])), errors="coerce")
    pred = pd.to_numeric(pd.Series(forecast.get("predictions", [])), errors="coerce")
    lower = pd.to_numeric(pd.Series(forecast.get("lower_ci", [])), errors="coerce")
    upper = pd.to_numeric(pd.Series(forecast.get("upper_ci", [])), errors="coerce")

    forecast_df = pd.DataFrame(
        {
            "date": fc_dates,
            "predicted": pred,
            "lower": lower,
            "upper": upper,
        }
    ).dropna(subset=["date", "predicted"])

    if not forecast_df.empty:
        forecast_df["horizon_days"] = np.arange(1, len(forecast_df) + 1)

    htable = pd.DataFrame(prediction.get("horizon_table", []))
    if not htable.empty and "horizon_days" in htable.columns and "forecast_volatility" in htable.columns:
        vol_map = (
            htable[["horizon_days", "forecast_volatility"]]
            .dropna(subset=["forecast_volatility"])
            .set_index("horizon_days")["forecast_volatility"]
            .to_dict()
        )
        if not forecast_df.empty:
            forecast_df["forecast_volatility"] = forecast_df["horizon_days"].map(vol_map)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_tail["Date"],
            y=hist_tail["Close"],
            mode="lines",
            name="Historical Close",
            line=dict(color="#4c566a", width=2),
            yaxis="y",
        )
    )

    if include_volume and "Volume" in hist_tail.columns:
        vol_series = pd.to_numeric(hist_tail["Volume"], errors="coerce").fillna(0.0)
        fig.add_trace(
            go.Bar(
                x=hist_tail["Date"],
                y=vol_series,
                name="Volume",
                marker_color="rgba(31, 119, 180, 0.22)",
                yaxis="y2",
                opacity=0.6,
            )
        )

    if not forecast_df.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["upper"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["lower"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(214, 39, 40, 0.12)",
                line=dict(width=0),
                name="Forecast CI (95%)",
                hoverinfo="skip",
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df["predicted"],
                mode="lines+markers",
                name="ARIMAX Forecast",
                line=dict(color="#d62728", width=3),
                marker=dict(size=6),
                yaxis="y",
            )
        )

        if "forecast_volatility" in forecast_df.columns and forecast_df["forecast_volatility"].notna().any():
            vol_vals = forecast_df["forecast_volatility"].fillna(0.0)
            vmax = max(float(vol_vals.max()), 1e-9)
            marker_sizes = 8 + ((vol_vals / vmax) * 16)
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["predicted"],
                    mode="markers",
                    name="GARCH Volatility Signal",
                    marker=dict(
                        size=marker_sizes,
                        color=vol_vals,
                        colorscale="Turbo",
                        showscale=True,
                        colorbar=dict(title="GARCH Vol"),
                        line=dict(width=0.8, color="#111111"),
                    ),
                    customdata=np.column_stack([forecast_df["horizon_days"], vol_vals]),
                    hovertemplate="Date=%{x}<br>Horizon=D+%{customdata[0]}<br>Pred=%{y:.2f}<br>GARCH Vol=%{customdata[1]:.6f}<extra></extra>",
                    yaxis="y",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=vol_vals,
                    mode="lines+markers",
                    name="GARCH Volatility Curve",
                    line=dict(color="#8c564b", width=2, dash="dot"),
                    marker=dict(size=5),
                    yaxis="y3",
                )
            )

    fig.update_layout(
        title="Smart Unified View: ARIMAX Forecast + CI + Volume + GARCH Volatility",
        template="plotly_white",
        hovermode="x unified",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price", side="left"),
        yaxis2=dict(
            title="Volume",
            side="right",
            overlaying="y",
            showgrid=False,
            rangemode="tozero",
        ),
        yaxis3=dict(
            title="GARCH Volatility",
            side="right",
            overlaying="y",
            showgrid=False,
            position=0.93,
        ),
        barmode="overlay",
    )
    return fig


def make_simple_final_smart_chart(details_df: pd.DataFrame, selected_horizon: int) -> go.Figure:
    chart_df = (
        details_df[details_df["horizon_days"] == selected_horizon]
        .sort_values("target_date")
        .copy()
    )
    fig = go.Figure()
    if chart_df.empty:
        return fig

    if "forecast_volatility" in chart_df.columns and chart_df["forecast_volatility"].notna().any():
        vol = chart_df["forecast_volatility"].fillna(0.0).astype(float)
        band = chart_df["predicted_price"].astype(float) * vol
    else:
        fallback_pct = max(float(chart_df["abs_pct_error"].mean()), 0.5) / 100.0
        band = chart_df["predicted_price"].astype(float) * fallback_pct

    chart_df["pred_plus"] = chart_df["predicted_price"].astype(float) + band
    chart_df["pred_minus"] = chart_df["predicted_price"].astype(float) - band

    fig.add_trace(
        go.Scatter(
            x=chart_df["target_date"],
            y=chart_df["actual_price"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["target_date"],
            y=chart_df["predicted_price"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["target_date"],
            y=chart_df["pred_plus"],
            mode="lines",
            name="Predicted + Band",
            line=dict(color="#ff9896", width=1.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["target_date"],
            y=chart_df["pred_minus"],
            mode="lines",
            name="Predicted - Band",
            line=dict(color="#ff9896", width=1.6),
        )
    )

    if "forecast_volatility" in chart_df.columns and chart_df["forecast_volatility"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=chart_df["target_date"],
                y=chart_df["forecast_volatility"].astype(float),
                mode="lines+markers",
                name="GARCH Volatility",
                line=dict(color="#9467bd", width=2, dash="dot"),
                yaxis="y2",
            )
        )

    fig.update_layout(
        title=f"Final Smart View (D+{selected_horizon}): Actual vs Predicted with Simple Bands",
        xaxis_title="Target Date",
        yaxis=dict(title="Price"),
        yaxis2=dict(title="GARCH Volatility", overlaying="y", side="right", showgrid=False),
        template="plotly_white",
        height=470,
        hovermode="x unified",
    )
    return fig




def run_historical_validation(
    history_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    horizons: list[int],
    auto_order: bool,
    manual_order: tuple[int, int, int],
    model_type: str,
    use_log_returns: bool = False,
    use_sentiment: bool = False,
    use_volume: bool = False,
    criterion: str = "aic",
    use_garch: bool = False,
    garch_distribution: str = "normal",
    rolling_window: bool = True,
    garch_log_returns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    valid_horizons = sorted({int(h) for h in horizons if int(h) > 0})
    if not valid_horizons:
        return pd.DataFrame(), pd.DataFrame()

    max_h = max(valid_horizons)
    candidate_idx = [
        idx
        for idx, dt in enumerate(df["Date"])
        if start_date <= dt <= end_date and (idx + max_h) < len(df)
    ]

    rows: list[dict] = []
    for idx in candidate_idx:
        train_df = df.iloc[: idx + 1].copy()
        if len(train_df) < 60:
            continue

        try:
            forecast_bundle = predict_series(
                train_df,
                horizons=valid_horizons,
                auto_order=auto_order,
                manual_order=manual_order,
                model_type=model_type,
                use_log_returns=use_log_returns,
                use_sentiment=use_sentiment,
                use_volume=use_volume,
                criterion=criterion,
                use_garch=use_garch,
                garch_distribution=garch_distribution,
                rolling_window=rolling_window,
                garch_log_returns=garch_log_returns,
            )
        except Exception:
            continue

        table = forecast_bundle["horizon_table"].set_index("horizon_days")
        base_price = float(df.iloc[idx]["Close"])
        base_date = df.iloc[idx]["Date"]

        for h in valid_horizons:
            if h not in table.index:
                continue

            target_idx = idx + h
            if target_idx >= len(df):
                continue

            predicted_price = float(table.loc[h, "predicted_price"])
            actual_price = float(df.iloc[target_idx]["Close"])
            forecast_volatility = np.nan
            if "forecast_volatility" in table.columns:
                raw_vol = table.loc[h, "forecast_volatility"]
                if pd.notna(raw_vol):
                    forecast_volatility = float(raw_vol)
            abs_error = abs(predicted_price - actual_price)
            sq_error = abs_error ** 2
            abs_pct_error = (abs_error / abs(actual_price) * 100.0) if actual_price else 0.0

            pred_change = predicted_price - base_price
            actual_change = actual_price - base_price
            direction_correct = (
                (pred_change > 0 and actual_change > 0)
                or (pred_change < 0 and actual_change < 0)
                or (pred_change == 0 and actual_change == 0)
            )

            rows.append(
                {
                    "cutoff_date": base_date,
                    "target_date": df.iloc[target_idx]["Date"],
                    "horizon_days": int(h),
                    "base_price": round(base_price, 4),
                    "predicted_price": round(predicted_price, 4),
                    "actual_price": round(actual_price, 4),
                    "abs_error": round(abs_error, 4),
                    "sq_error": round(sq_error, 4),
                    "abs_pct_error": round(abs_pct_error, 4),
                    "accuracy_pct": float(abs_pct_error <= 1.0) * 100.0,
                    "direction_correct": int(direction_correct),
                    "forecast_volatility": forecast_volatility,
                }
            )

    details = pd.DataFrame(rows)
    if details.empty:
        return pd.DataFrame(), pd.DataFrame()

    summary = (
        details.groupby("horizon_days", as_index=False)
        .agg(
            samples=("actual_price", "count"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: float((s.mean()) ** 0.5)),
            mape=("abs_pct_error", "mean"),
            accuracy_pct=("accuracy_pct", "mean"),
            direction_accuracy_pct=("direction_correct", lambda s: float(s.mean()) * 100.0),
            avg_forecast_volatility=("forecast_volatility", "mean"),
        )
        .sort_values("horizon_days")
    )

    if details["forecast_volatility"].notna().any():
        garch_rows: list[dict] = []
        for h, gdf in details.groupby("horizon_days"):
            vg = gdf.dropna(subset=["forecast_volatility"])
            if vg.empty:
                continue

            high_cut = float(vg["forecast_volatility"].quantile(0.7))
            high_vol = vg[vg["forecast_volatility"] >= high_cut]
            high_vol_hit_rate = float(high_vol["accuracy_pct"].mean()) if not high_vol.empty else np.nan

            corr = np.nan
            if len(vg) >= 2 and float(vg["forecast_volatility"].std()) > 0:
                corr = float(vg["forecast_volatility"].corr(vg["abs_pct_error"]))

            garch_rows.append(
                {
                    "horizon_days": int(h),
                    "high_vol_hit_rate_pct": high_vol_hit_rate,
                    "vol_error_corr": corr,
                }
            )

        if garch_rows:
            garch_summary = pd.DataFrame(garch_rows)
            summary = summary.merge(garch_summary, on="horizon_days", how="left")

    for col in [
        "mae",
        "rmse",
        "mape",
        "accuracy_pct",
        "direction_accuracy_pct",
        "avg_forecast_volatility",
        "high_vol_hit_rate_pct",
        "vol_error_corr",
    ]:
        if col not in summary.columns:
            continue
        summary[col] = summary[col].round(3)

    return summary, details


def make_validation_chart(details_df: pd.DataFrame, selected_horizon: int) -> go.Figure:
    horizon_df = (
        details_df[details_df["horizon_days"] == selected_horizon]
        .sort_values("cutoff_date")
        .copy()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=horizon_df["target_date"],
            y=horizon_df["actual_price"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=horizon_df["target_date"],
            y=horizon_df["predicted_price"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Historical Validation (D+{selected_horizon})",
        xaxis_title="Target Date",
        yaxis_title="Price",
        template="plotly_white",
        height=450,
        hovermode="x unified",
    )
    return fig


def make_garch_validation_diagnostic_chart(details_df: pd.DataFrame, selected_horizon: int) -> go.Figure:
    horizon_df = (
        details_df[details_df["horizon_days"] == selected_horizon]
        .sort_values("target_date")
        .copy()
    )
    if "forecast_volatility" not in horizon_df.columns:
        return go.Figure()

    horizon_df = horizon_df.dropna(subset=["forecast_volatility"])
    if horizon_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=horizon_df["target_date"],
            y=horizon_df["forecast_volatility"],
            name="Forecast Volatility",
            marker_color="#9467bd",
            yaxis="y",
            opacity=0.65,
            hovertemplate="Date=%{x}<br>Vol=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=horizon_df["target_date"],
            y=horizon_df["abs_pct_error"],
            mode="lines+markers",
            name="Absolute % Error",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
            yaxis="y2",
            hovertemplate="Date=%{x}<br>Abs %% Error=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"GARCH Risk vs Error Diagnostics (D+{selected_horizon})",
        xaxis_title="Target Date",
        yaxis=dict(title="Forecast Volatility"),
        yaxis2=dict(title="Absolute % Error", overlaying="y", side="right"),
        template="plotly_white",
        height=430,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def make_validation_metric_chart(summary_df: pd.DataFrame) -> go.Figure:
    chart_df = summary_df.sort_values("horizon_days").copy()
    chart_df["label"] = chart_df["horizon_days"].apply(lambda h: f"D+{int(h)}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_df["label"],
            y=chart_df["accuracy_pct"],
            name="Hit Rate (Error ≤ 1%)",
            marker_color="#1f77b4",
            text=[f"{v:.1f}%" for v in chart_df["accuracy_pct"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=chart_df["label"],
            y=chart_df["direction_accuracy_pct"],
            name="Direction Accuracy (%)",
            marker_color="#ff7f0e",
            text=[f"{v:.1f}%" for v in chart_df["direction_accuracy_pct"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Validation Reliability By Horizon",
        xaxis_title="Horizon",
        yaxis_title="Percentage (%)",
        barmode="group",
        template="plotly_white",
        height=430,
        hovermode="x unified",
    )
    return fig


def make_validation_window_chart(
    history_window_df: pd.DataFrame,
    details_df: pd.DataFrame,
    selected_horizon: int,
) -> go.Figure:
    hist_df = history_window_df.copy().sort_values("Date")
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])

    horizon_df = (
        details_df[details_df["horizon_days"] == selected_horizon]
        .sort_values("target_date")
        .copy()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Close"],
            mode="lines",
            name="Past Close",
            line=dict(color="#6c757d", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=horizon_df["target_date"],
            y=horizon_df["actual_price"],
            mode="lines+markers",
            name="Past Actual",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=horizon_df["target_date"],
            y=horizon_df["predicted_price"],
            mode="lines+markers",
            name="Past Predicted",
            line=dict(color="#d62728", width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Same Chart View: Past Close + Past Predicted vs Past Actual (D+{selected_horizon})",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )
    return fig


def explain_validation_results(summary_df: pd.DataFrame, selected_horizon: int) -> str:
    row = summary_df[summary_df["horizon_days"] == selected_horizon]
    if row.empty:
        return "No validation result available for the selected horizon."

    r = row.iloc[0]
    quality = "strong" if r["accuracy_pct"] >= 50 else "moderate" if r["accuracy_pct"] >= 30 else "weak"
    direction_quality = "reliable" if r["direction_accuracy_pct"] >= 55 else "inconsistent"

    base_text = (
        f"D+{int(r['horizon_days'])} plain-language summary: "
        f"returns within a 1% error margin {r['accuracy_pct']:.2f}% of the time ({quality}), "
        f"average absolute error is {r['mae']:.2f}, and directional trend correctly predicted "
        f"{r['direction_accuracy_pct']:.2f}% of the time ({direction_quality}) over {int(r['samples'])} past samples."
    )

    if "high_vol_hit_rate_pct" in summary_df.columns and pd.notna(r.get("high_vol_hit_rate_pct", np.nan)):
        corr = r.get("vol_error_corr", np.nan)
        corr_text = "not enough data"
        if pd.notna(corr):
            corr_text = f"{float(corr):.2f}"
        base_text += (
            f" GARCH context: in high-volatility windows, hit rate is {float(r['high_vol_hit_rate_pct']):.2f}%, "
            f"and volatility-error correlation is {corr_text}."
        )

    return base_text


def explain_model_parameters(model_type: str, order: tuple[int, int, int], use_log: bool, use_sentiment: bool, use_volume: bool = False) -> str:
    p, d, q = order
    model_name = model_type.upper()

    lines = [
        f"{model_name}({p}, {d}, {q}) means:",
        f"- p={p}: looks back {p} previous day(s) to detect price momentum patterns.",
    ]
    
    if use_log:
        lines.append("- Log Returns: the model learns percentage rate of change instead of raw price levels, adjusting for exponential scaling automatically.")

    lines.append(f"- d={d}: applies differencing {d} time(s) to remove trend and stabilize the series.")
    lines.append(f"- q={q}: uses {q} previous forecast-error term(s) to correct prediction bias.")

    if model_type.lower() == "arimax":
        lines.append(
            "- X (exogenous features): also uses external signals from price behavior "
            "(daily return, 5-day moving average, 5-day volatility, 3-day momentum)."
        )
        if use_sentiment:
            lines.append("- Sentiment Feature: injects simulated market sentiment text/news scores as a leading correlation indicator.")
        if use_volume:
            lines.append("- Volume Feature: injects actual traded Volume data dynamically scaling predictions with market activity.")


    if d == 1:
        lines.append("In plain language: the model predicts day-to-day price change, then converts it back to price level.")
    else:
        lines.append("In plain language: the model predicts directly on transformed price dynamics from historical data.")

    return "\n".join(lines)


def get_validation_history_window(
    history_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    max_horizon: int,
) -> pd.DataFrame:
    df = history_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    eligible = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)]
    if eligible.empty:
        return pd.DataFrame(columns=["Date", "Close"])

    start_idx = int(eligible.index.min())
    cutoff_end_idx = int(eligible.index.max())
    target_end_idx = min(cutoff_end_idx + max_horizon, len(df) - 1)

    return df.iloc[start_idx : target_end_idx + 1][["Date", "Close"]].copy()


with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Select Backend Source:", ["PostgreSQL Database", "VNStock API (No DB required)"], index=1)

if "PostgreSQL" in data_source:
    try:
        repo = get_postgres_repo()
        db_info = repo.test_connection()
    except Exception as exc:
        st.error("Cannot connect to PostgreSQL. Please check credentials or switch to VNStock API Data Source in the sidebar.")
        st.exception(exc)
        st.stop()
else:
    repo = get_api_repo()
    db_info = repo.test_connection()

companies_df = load_companies(repo, data_source)
symbols = sorted(companies_df["symbol"].astype(str).str.upper().unique().tolist()) if not companies_df.empty else []
sectors = sorted(companies_df["sector"].astype(str).unique().tolist()) if not companies_df.empty else []

with st.sidebar:
    st.header("Database")
    st.success(
        f"Connected to {db_info['database']} ({repo.get_schema_mode()} schema)",
        icon="✅",
    )
    st.caption(db_info["version"])

    target_mode = st.radio("Prediction target", ["Company", "Sector Group"], index=0)
    selected_model_type = st.selectbox(
        "Model type",
        options=["arima", "arimax"],
        index=0,
        help="Choose ARIMA or ARIMAX (ARIMA with exogenous features).",
    )
    st.session_state["selected_model_type"] = selected_model_type

    information_criterion = st.radio(
        "Information Criterion",
        options=["AIC", "BIC"],
        index=0,
        help="This controls auto model selection for both ARIMA/ARIMAX and Auto-GARCH.",
    )
    selected_criterion = information_criterion.lower()

    use_garch = st.toggle(
        "ARIMAX + Auto-GARCH (Rolling Window)",
        value=False,
        help="Enable ARIMAX residual-volatility modeling with Auto-GARCH search.",
    )

    rolling_window = st.toggle(
        "Use Rolling Window Validation",
        value=True,
        disabled=use_garch,
        help="Required and auto-locked when GARCH mode is enabled.",
    )
    if use_garch:
        rolling_window = True
        st.caption("Rolling window is required for GARCH mode and has been locked ON.")

    garch_log_returns = st.toggle(
        "GARCH Pipeline: Use Log Returns",
        value=True,
        help="When enabled, ARIMAX+GARCH runs on log returns for stable volatility modeling.",
    )

    garch_distribution = "normal"
    if use_garch:
        garch_distribution = st.radio(
            "GARCH Distribution",
            options=["Gaussian (Normal)", "Student's t-distribution"],
            index=0,
            help="Distribution assumption used by the GARCH residual model.",
        )
    
    use_log_returns = st.toggle("Model Log Returns (Stationarity)", value=False, help="Model stock price returns (log returns) rather than absolute price levels to improve stationarity.")
    st.session_state["use_log_returns"] = use_log_returns
    
    use_sentiment = False
    use_volume = False
    if selected_model_type == "arimax":
        use_sentiment = st.toggle("Include Sentiment Feature", value=False, help="Integrate market news sentiment as a daily exogenous feature in ARIMAX.")
        st.session_state["use_sentiment"] = use_sentiment
        
        use_volume = st.toggle("Include Actual Trade Volume", value=False, help="Integrate raw trading Volume from your database as an exogenous feature in ARIMAX.")
        st.session_state["use_volume"] = use_volume
    selected_days = st.multiselect(
        "Select analysis horizons (days)",
        options=[1, 3, 5],
        default=[1, 3, 5],
        help="Trading-day horizons to analyze: 1 day, 3 days, and 5 days.",
    )

    auto_order = st.toggle("Auto-select ARIMA order (recommended)", value=True)
    if not auto_order:
        p = st.number_input("p", min_value=0, max_value=5, value=1)
        d = st.number_input("d", min_value=0, max_value=2, value=1)
        q = st.number_input("q", min_value=0, max_value=5, value=1)
        manual_order = (int(p), int(d), int(q))
    else:
        manual_order = (1, 1, 1)

if not selected_days:
    st.warning("Select at least one analysis horizon in the sidebar.")
    st.stop()

if target_mode == "Company":
    if not symbols:
        if "PostgreSQL" in data_source:
            st.warning("No companies found in database.")
            st.info("Click below to import VN100 stock data from VNStock API directly into your PostgreSQL database.")
            if st.button("🚀 Import VN100 Data to Database", type="primary", use_container_width=True):
                try:
                    from Import_Data import (
                        get_default_symbols,
                        import_symbols_to_db,
                        configure_vnstock_api_key,
                        load_env_file as import_load_env,
                    )

                    import_load_env()
                    configure_vnstock_api_key()

                    import_symbols = get_default_symbols()
                    if not import_symbols:
                        st.error("VN100.txt not found. Please create VN100.txt with stock symbols.")
                        st.stop()

                    st.write(f"Importing **{len(import_symbols)}** symbols into PostgreSQL...")
                    progress_bar = st.progress(0, text="Starting import...")
                    status_container = st.container()
                    counts = {"ok": 0, "error": 0, "skip": 0}

                    conn = repo._connect()
                    try:
                        def ui_progress(idx, total, result):
                            pct = (idx + 1) / total
                            if result["status"] == "ok":
                                counts["ok"] += 1
                            elif result["status"] == "error":
                                counts["error"] += 1
                            else:
                                counts["skip"] += 1
                            progress_bar.progress(
                                pct,
                                text=f"[{idx+1}/{total}] {result['symbol']}: {result['status'].upper()} – {result['message']}"
                            )

                        results = import_symbols_to_db(
                            import_symbols,
                            conn,
                            legacy_tables=True,
                            progress_callback=ui_progress,
                        )
                    finally:
                        conn.close()

                    progress_bar.progress(1.0, text="Import complete!")
                    status_container.success(
                        f"✅ Import finished: {counts['ok']} OK, {counts['skip']} skipped, {counts['error']} errors"
                    )
                    if counts["error"] > 0:
                        failed = [r for r in results if r["status"] == "error"]
                        with status_container.expander("Show errors"):
                            for r in failed:
                                st.write(f"**{r['symbol']}**: {r['message']}")

                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}")
                    st.exception(exc)
            st.stop()
        else:
            st.warning("No companies found. Check VN100.txt or switch to PostgreSQL mode.")
            st.stop()

    if sectors:
        selected_sector = st.selectbox("Filter by sector", ["All"] + sectors)
        if selected_sector == "All":
            candidate_symbols = symbols
        else:
            candidate_symbols = (
                companies_df[companies_df["sector"] == selected_sector]["symbol"]
                .astype(str)
                .str.upper()
                .sort_values()
                .tolist()
            )
    else:
        selected_sector = "All"
        candidate_symbols = symbols

    selected_target = st.selectbox("Choose company symbol", candidate_symbols)
    history_df = load_symbol_history(repo, data_source, selected_target)
    chart_title = f"{selected_target} - Historical vs Forecast"
else:
    if not sectors:
        st.warning("No sector information in company_info table. Run Import_Data.py first.")
        st.stop()

    selected_target = st.selectbox("Choose sector group", sectors)
    selected_sector = selected_target
    history_df = load_sector_history(repo, data_source, selected_target)
    chart_title = f"Sector Average ({selected_target}) - Historical vs Forecast"

if history_df.empty:
    st.error("Not enough data for this target. Need at least 60 data points.")
    st.stop()


st.subheader("Data & Analysis Configuration")
metric_col, date_col, cap_col = st.columns(3)
with metric_col:
    st.write(f"**Target:** {selected_target}")
    st.write(f"**Rows available:** {len(history_df)}")
    st.write(f"**Range:** {history_df['Date'].min().date()} to {history_df['Date'].max().date()}")

    # Data Continuity Check
    dates = pd.to_datetime(history_df["Date"]).sort_values()
    diffs = dates.diff().dt.days.dropna()
    gaps = diffs[diffs > 3]
    if not gaps.empty:
        st.warning(f"⚠️ **Data Quality Warning:** Found {len(gaps)} significant time gaps in history (e.g. missing >3 days). Time series models assume regular intervals. Consider checking the data source or ensuring these are only standard market holidays.")

# Setup validation parameters
validation_df = history_df.copy().sort_values("Date")
validation_df["Date"] = pd.to_datetime(validation_df["Date"])
hist_min = validation_df["Date"].min()
hist_max = validation_df["Date"].max()

default_end = hist_max - pd.Timedelta(days=max(selected_days) + 2)
if default_end < hist_min:
    default_end = hist_min
default_start = max(hist_min, default_end - pd.Timedelta(days=120))

with date_col:
    backtest_range = st.date_input(
        "Validation Backtest Range",
        value=(hist_min.date(), default_end.date()),
        min_value=hist_min.date(),
        max_value=hist_max.date(),
    )
    validation_horizons = st.multiselect(
        "Validation horizons",
        options=selected_days,
        default=selected_days,
        help="Use the same horizons you forecast, such as 1/3/5.",
    )

with cap_col:
    initial_capital = st.number_input(
        "Initial capital ($)",
        min_value=1000.0,
        value=100000.0,
        step=1000.0,
    )

# Preview Backtest Range
if isinstance(backtest_range, (list, tuple)) and len(backtest_range) == 2:
    month_start_ts = pd.Timestamp(backtest_range[0])
    month_end_ts = pd.Timestamp(backtest_range[1])
    if month_start_ts <= month_end_ts:
        with st.expander("Preview Backtest Date Range in History", expanded=False):
            month_fig = make_month_window_chart(validation_df, month_start_ts, month_end_ts)
            st.plotly_chart(month_fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
run_full_analysis = st.button("Run Full Historical Diagnostics & Backtest", type="primary", use_container_width=True)

if run_full_analysis:
    if not isinstance(backtest_range, (list, tuple)) or len(backtest_range) != 2:
        st.warning("Please select both start and end dates for validation.")
    elif not validation_horizons:
        st.warning("Please select at least one validation horizon.")
    else:
        start_ts = pd.Timestamp(backtest_range[0])
        end_ts = pd.Timestamp(backtest_range[1])
        if start_ts > end_ts:
            st.warning("Validation start date must be before end date.")
        else:
            with st.spinner("Phase 1: Training ARIMA on full dataset for diagnostics..."):
                prediction = predict_series(
                    history_df,
                    horizons=selected_days,
                    auto_order=auto_order,
                    manual_order=manual_order,
                    model_type=selected_model_type,
                    use_log_returns=use_log_returns,
                    use_sentiment=use_sentiment,
                    use_volume=use_volume,
                    criterion=selected_criterion,
                    use_garch=use_garch,
                    garch_distribution=garch_distribution,
                    rolling_window=rolling_window,
                    garch_log_returns=garch_log_returns,
                )
                st.session_state["prediction_diagnostics"] = prediction

            with st.spinner("Phase 2: Running historical backtest over selected past dates..."):
                val_summary, val_details = run_historical_validation(
                    validation_df,
                    start_date=start_ts,
                    end_date=end_ts,
                    horizons=validation_horizons,
                    auto_order=auto_order,
                    manual_order=manual_order,
                    model_type=selected_model_type,
                    use_log_returns=use_log_returns,
                    use_sentiment=use_sentiment,
                    use_volume=use_volume,
                    criterion=selected_criterion,
                    use_garch=use_garch,
                    garch_distribution=garch_distribution,
                    rolling_window=rolling_window,
                    garch_log_returns=garch_log_returns,
                )
                val_history_used = get_validation_history_window(
                    validation_df,
                    start_date=start_ts,
                    end_date=end_ts,
                    max_horizon=max(validation_horizons),
                )
                st.session_state["val_summary"] = val_summary
                st.session_state["val_details"] = val_details
                st.session_state["val_history_used"] = val_history_used

if "prediction_diagnostics" in st.session_state:
    st.markdown("---")
    st.subheader("Phase 1: Historical Model Fit & Diagnostics")
    
    prediction = st.session_state["prediction_diagnostics"]
    eval_metrics = prediction["evaluation"]
    stationarity = prediction["stationarity"]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Selected Order", str(prediction["selected_order"]))
    metric_cols[1].metric("RMSE", f"{eval_metrics['rmse']:.2f}")
    metric_cols[2].metric("MAE", f"{eval_metrics['mae']:.2f}")
    metric_cols[3].metric("MAPE", f"{eval_metrics['mape']:.2f}%")

    garch_block = prediction.get("garch", {})
    if garch_block.get("enabled"):
        gcols = st.columns(3)
        gcols[0].metric("GARCH", "Enabled")
        selected_garch_order = garch_block.get("selected_order") or {}
        gcols[1].metric("GARCH Order", f"({selected_garch_order.get('p', '-')},{selected_garch_order.get('q', '-')})")
        gcols[2].metric("GARCH Dist", str(garch_block.get("distribution", "normal")).upper())
        st.caption(garch_block.get("reason", "Auto-GARCH completed."))

        htable = pd.DataFrame(prediction.get("horizon_table", []))
        trials_df = pd.DataFrame(garch_block.get("trials", []))
        criterion_col = str(garch_block.get("criterion", prediction.get("criterion", "aic"))).lower()
        if "forecast_volatility" in htable.columns and not htable.empty:
            st.markdown("#### GARCH Interactive Visualization")
            g_tabs = st.tabs(["Price + Volatility Overlay", "Volatility Curve", "Auto-GARCH Search"])

            with g_tabs[0]:
                overlay_fig = make_garch_forecast_overlay_chart(htable)
                st.plotly_chart(overlay_fig, use_container_width=True)

                selected_h = st.select_slider(
                    "Inspect horizon",
                    options=htable["horizon_days"].astype(int).tolist(),
                    value=int(htable["horizon_days"].astype(int).iloc[0]),
                )
                selected_row = htable[htable["horizon_days"] == selected_h].iloc[0]
                hcols = st.columns(4)
                hcols[0].metric("Horizon", f"D+{int(selected_h)}")
                hcols[1].metric("Predicted Price", f"{float(selected_row['predicted_price']):.2f}")
                hcols[2].metric("Volatility", f"{float(selected_row['forecast_volatility']):.6f}")
                hcols[3].metric("Trend", str(selected_row.get("trend", "N/A")))

            with g_tabs[1]:
                vol_fig = make_garch_volatility_term_structure_chart(htable)
                st.plotly_chart(vol_fig, use_container_width=True)

            with g_tabs[2]:
                if not trials_df.empty:
                    heatmap_fig = make_garch_search_heatmap(trials_df, criterion_col)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    st.dataframe(trials_df.sort_values(["p", "q"]), use_container_width=True)
                else:
                    st.info("No Auto-GARCH trial diagnostics were returned.")

    st.caption(f"Model type used: {prediction.get('model_type', selected_model_type).upper()}")
    st.info(explain_model_parameters(selected_model_type, prediction["selected_order"], use_log_returns, use_sentiment, use_volume))

    st.caption(stationarity["interpretation"])

    order_diag_df = pd.DataFrame(prediction.get("order_diagnostics", []))
    if auto_order and not order_diag_df.empty:
        st.info(prediction.get("order_selection_reason", "Auto-order selected from tested PDQ candidates."))
        diag_fig = make_order_diagnostic_chart(order_diag_df, prediction["selected_order"])
        st.plotly_chart(diag_fig, use_container_width=True)

        pdq_fig = make_pdq_map_chart(order_diag_df, prediction["selected_order"])
        st.plotly_chart(pdq_fig, use_container_width=True)

        failed = order_diag_df[order_diag_df["status"] != "ok"]
        if not failed.empty:
            st.caption("Some candidate orders failed to converge during auto-selection.")

    acf_vals, pacf_vals = get_acf_pacf(history_df)
    if acf_vals and pacf_vals:
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF"))
        fig_acf.add_trace(go.Scatter(x=list(range(len(pacf_vals))), y=pacf_vals, mode="markers", name="PACF", marker=dict(size=8, color="red")))
        fig_acf.update_layout(
            title="Auto-ARIMA Diagnostic: ACF & PACF on Differenced Returns",
            xaxis_title="Lag",
            yaxis_title="Correlation",
            height=300,
            template="plotly_white"
        )
        st.write("Why did Auto-ARIMA pick those P and Q values? High bars in ACF (blue) indicate moving average (Q) lags, while high PACF (red dots) indicate autoregressive (P) lags needed.")
        st.plotly_chart(fig_acf, use_container_width=True)

    st.text(prediction["model_summary"]["full_summary"])


if "val_summary" in st.session_state and "val_details" in st.session_state:
    st.markdown("---")
    st.subheader("Phase 2: Historical Validation & Backtest Results")
    
    val_summary = st.session_state["val_summary"]
    val_details = st.session_state["val_details"]
    if val_summary.empty:
        st.info("No validation samples available in this range. Try a wider range.")
    else:
        metric_cols = st.columns(len(val_summary))
        for idx, row in enumerate(val_summary.itertuples(index=False)):
            with metric_cols[idx]:
                st.metric(
                    label=f"D+{row.horizon_days} Hit Rate (<1% Err)",
                    value=f"{row.accuracy_pct:.2f}%",
                    delta=f"Dir {row.direction_accuracy_pct:.2f}%",
                )
                if hasattr(row, "avg_forecast_volatility") and pd.notna(row.avg_forecast_volatility):
                    st.caption(f"Avg Vol {row.avg_forecast_volatility:.6f}")
                if hasattr(row, "high_vol_hit_rate_pct") and pd.notna(row.high_vol_hit_rate_pct):
                    st.caption(f"High-Vol Hit {row.high_vol_hit_rate_pct:.2f}%")

        metric_fig = make_validation_metric_chart(val_summary)
        st.plotly_chart(metric_fig, use_container_width=True)

        chart_horizon = st.selectbox(
            "Validation chart horizon",
            options=val_summary["horizon_days"].astype(int).tolist(),
            index=0,
        )
        st.info(explain_validation_results(val_summary, chart_horizon))

        val_fig = make_validation_chart(val_details, chart_horizon)
        st.plotly_chart(val_fig, use_container_width=True)

        horizon_row = val_summary[val_summary["horizon_days"] == chart_horizon]
        if not horizon_row.empty and "high_vol_hit_rate_pct" in horizon_row.columns:
            hv = horizon_row.iloc[0].get("high_vol_hit_rate_pct", np.nan)
            corr = horizon_row.iloc[0].get("vol_error_corr", np.nan)
            avg_vol = horizon_row.iloc[0].get("avg_forecast_volatility", np.nan)
            if pd.notna(hv) or pd.notna(corr) or pd.notna(avg_vol):
                st.markdown("#### GARCH Validation Diagnostics")
                gcols = st.columns(3)
                gcols[0].metric(
                    "High-Vol Hit Rate (<1% Err)",
                    f"{float(hv):.2f}%" if pd.notna(hv) else "N/A",
                )
                gcols[1].metric(
                    "Volatility-Error Correlation",
                    f"{float(corr):.2f}" if pd.notna(corr) else "N/A",
                )
                gcols[2].metric(
                    "Average Forecast Volatility",
                    f"{float(avg_vol):.6f}" if pd.notna(avg_vol) else "N/A",
                )

                garch_val_fig = make_garch_validation_diagnostic_chart(val_details, chart_horizon)
                if garch_val_fig.data:
                    st.plotly_chart(garch_val_fig, use_container_width=True)

        if "val_history_used" in st.session_state:
            combined_fig = make_validation_window_chart(
                st.session_state["val_history_used"],
                val_details,
                chart_horizon,
            )
            st.plotly_chart(combined_fig, use_container_width=True)

        profit_summary, profit_df = compute_profit_projection(
            val_details,
            selected_horizon=chart_horizon,
            initial_capital=initial_capital,
        )
        if profit_summary:
            st.subheader("Return & Profit Analytics")
            pcols = st.columns(4)
            pcols[0].metric("Average Trade Return", f"{profit_summary['avg_return_pct']:.2f}%")
            pcols[1].metric("Cumulative Return", f"{profit_summary['cumulative_return_pct']:.2f}%")
            pcols[2].metric("Annualized Return", f"{profit_summary['annualized_return_pct']:.2f}%")
            pcols[3].metric("Projected 1Y Profit", f"{profit_summary['projected_year_profit_value']:,.2f}")

            st.info(
                f"Plain language: based on D+{chart_horizon} validation trades, the strategy returned "
                f"{profit_summary['avg_return_pct']:.2f}% on average per trade and projects about "
                f"{profit_summary['projected_year_profit_value']:,.2f} profit over one year on "
                f"starting capital {initial_capital:,.2f}."
            )

            profit_fig = make_profit_curve_chart(profit_df)
            st.plotly_chart(profit_fig, use_container_width=True)

        st.markdown("#### Final Smart Chart View")
        st.caption("Simple view: Actual, Predicted, Predicted + Band, Predicted - Band, with GARCH volatility overlay when available.")
        final_smart_fig = make_simple_final_smart_chart(val_details, chart_horizon)
        st.plotly_chart(final_smart_fig, use_container_width=True)
