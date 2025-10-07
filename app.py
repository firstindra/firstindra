"""Streamlit dashboard for real-time stock and crypto price prediction using LSTM."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="LSTM Real-Time Price Dashboard", layout="wide")


@dataclass
class ModelConfig:
    ticker: str
    look_back: int
    epochs: int
    batch_size: int
    prediction_horizon: int
    interval: str
    period: str


def _get_default_period(interval: str) -> str:
    """Map interval to default history period."""
    if interval in {"1m", "2m", "5m", "15m", "30m"}:
        return "7d"
    if interval in {"60m", "90m", "1h"}:
        return "60d"
    return "5y"


@st.cache_data(show_spinner=False)
def load_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Load price data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("Ticker tidak ditemukan atau data tidak tersedia untuk kombinasi periode dan interval tersebut.")
    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "timestamp"
    return df


def create_sequences(data: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert array to sequences for LSTM."""
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i : i + look_back, 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


@lru_cache(maxsize=8)
def train_model(cache_key: str, config: ModelConfig, scaled_values: Tuple[float, ...]) -> Tuple[tf.keras.Model, MinMaxScaler]:
    """Train LSTM model; cached by ticker and parameters to avoid retraining unnecessarily."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(scaled_values).reshape(-1, 1))
    x, y = create_sequences(scaled_data, config.look_back)
    if len(x) == 0:
        raise ValueError("Data terlalu sedikit untuk membuat sequence. Coba kurangi look back atau pilih interval yang lebih rapat.")

    train_size = int(len(x) * 0.8)
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:], y[train_size:]

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))

    model = build_lstm_model((config.look_back, 1))
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val) if len(x_val) > 0 else None,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
    )
    return model, scaler


def iterative_forecast(
    model: tf.keras.Model,
    scaler: MinMaxScaler,
    history: np.ndarray,
    look_back: int,
    horizon: int,
) -> List[float]:
    """Generate iterative predictions for a given horizon."""
    forecast_input = history[-look_back:].reshape(1, look_back, 1)
    predictions = []
    for _ in range(horizon):
        pred_scaled = model.predict(forecast_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0, 0]
        predictions.append(pred)

        new_input = np.append(forecast_input[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
        forecast_input = new_input
    return predictions


def display_metrics(actual: np.ndarray, predicted: np.ndarray) -> None:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    metric_columns = st.columns(3)
    metric_columns[0].metric("RMSE", f"{rmse:,.4f}")
    metric_columns[1].metric("MAE", f"{mae:,.4f}")
    metric_columns[2].metric("MAPE", f"{mape:,.2f}%")


def main() -> None:
    st.title("Dashboard Prediksi Harga Saham & Kripto Real-Time")
    st.caption("Menggunakan model LSTM dengan data Yahoo Finance yang diperbarui secara real-time.")

    st.sidebar.header("Pengaturan Model")
    asset_type = st.sidebar.selectbox("Jenis Aset", ["Saham", "Kripto"])
    default_ticker = "AAPL" if asset_type == "Saham" else "BTC-USD"
    ticker = st.sidebar.text_input("Ticker", default_ticker).upper()

    interval = st.sidebar.selectbox(
        "Interval", ["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"], index=5
    )
    period = st.sidebar.selectbox(
        "Periode", [
            "7d",
            "30d",
            "90d",
            "180d",
            "1y",
            "2y",
            "5y",
            "10y",
        ],
        index=["7d", "30d", "90d", "180d", "1y", "2y", "5y", "10y"].index(_get_default_period(interval)),
    )

    look_back = st.sidebar.slider("Look Back (jumlah candle)", min_value=10, max_value=200, value=60, step=5)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    horizon = st.sidebar.slider("Horizon Prediksi", min_value=1, max_value=60, value=10)

    refresh = st.sidebar.button("Perbarui Data")

    try:
        if refresh:
            load_price_data.clear()
        with st.spinner("Mengambil data harga..."):
            price_df = load_price_data(ticker, period, interval)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.subheader(f"Harga Penutupan {ticker}")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=price_df.index, y=price_df["close"], name="Harga Penutupan"))
    fig_price.update_layout(height=400, xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(fig_price, use_container_width=True)

    config = ModelConfig(
        ticker=ticker,
        look_back=look_back,
        epochs=epochs,
        batch_size=batch_size,
        prediction_horizon=horizon,
        interval=interval,
        period=period,
    )

    scaled_values = tuple(price_df["close"].astype(float).values)
    cache_key = "|".join(
        [
            ticker,
            interval,
            period,
            str(look_back),
            str(epochs),
            str(batch_size),
        ]
    )

    try:
        with st.spinner("Melatih model LSTM..."):
            model, scaler = train_model(cache_key, config, scaled_values)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    scaled_close = scaler.transform(price_df[["close"]])
    x, y = create_sequences(scaled_close, look_back)
    x = x.reshape((x.shape[0], x.shape[1], 1))
    predictions_scaled = model.predict(x, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled).reshape(-1)

    actual = price_df["close"].values[look_back:]
    aligned_index = price_df.index[look_back:]

    st.subheader("Perbandingan Harga Aktual vs Prediksi")
    comparison_fig = go.Figure()
    comparison_fig.add_trace(go.Scatter(x=aligned_index, y=actual, name="Aktual"))
    comparison_fig.add_trace(go.Scatter(x=aligned_index, y=predictions, name="Prediksi"))
    comparison_fig.update_layout(height=400, xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(comparison_fig, use_container_width=True)

    display_metrics(actual, predictions)

    st.subheader("Prediksi Harga ke Depan")
    future_predictions = iterative_forecast(
        model,
        scaler,
        scaled_close,
        look_back,
        horizon,
    )
    last_timestamp = price_df.index[-1]

    if isinstance(last_timestamp, dt.datetime):
        if interval.endswith("m"):
            minutes = int(interval.rstrip("m"))
            delta = dt.timedelta(minutes=minutes)
        elif interval.endswith("h"):
            hours = int(interval.rstrip("h"))
            delta = dt.timedelta(hours=hours)
        elif interval == "1d":
            delta = dt.timedelta(days=1)
        elif interval == "1wk":
            delta = dt.timedelta(weeks=1)
        elif interval == "1mo":
            delta = dt.timedelta(days=30)
        else:
            delta = dt.timedelta(days=1)
    else:
        delta = dt.timedelta(days=1)

    future_index = [last_timestamp + (i + 1) * delta for i in range(horizon)]
    forecast_df = pd.DataFrame({"Prediksi": future_predictions}, index=future_index)

    future_fig = go.Figure()
    future_fig.add_trace(
        go.Scatter(
            x=list(price_df.index[-horizon:]) + future_index,
            y=list(price_df["close"].values[-horizon:]) + future_predictions,
            name="Prediksi ke Depan",
        )
    )
    future_fig.update_layout(height=400, xaxis_title="Tanggal", yaxis_title="Harga")
    st.plotly_chart(future_fig, use_container_width=True)

    st.dataframe(forecast_df.style.format("{:.2f}"))

    st.markdown(
        """
        **Catatan**:

        - Model LSTM dilatih ulang setiap kali parameter diubah.
        - Gunakan tombol *Perbarui Data* untuk menarik harga terbaru dari Yahoo Finance.
        - Untuk data intraday (1m-30m) Yahoo Finance hanya menyediakan historis sekitar 7 hari.
        - Prediksi bersifat eksperimental dan tidak dapat dijadikan dasar keputusan investasi.
        """
    )


if __name__ == "__main__":
    main()
