import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
import plotly.express as px 
import ta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Data fetch
# -------------------------------------------------------------------
def fetch_data(ticker: str) -> pd.DataFrame:
    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)

    # yfinance sometimes returns multiindex columns; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# ============================================================
# 📌 FEATURE ENGINEERING
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- base OHLCV assumed: Close, High, Low, Open, Volume ---
    # Technical indicators (all based on Close & Volume)
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(20).std()

    # Optional: keep 7-day mean as a FEATURE (not target)
    df["SMA_7"] = df["Close"].rolling(7).mean()

    # ✅ CHANGED: target is now REAL NEXT-DAY CLOSE, not 7-day average
    df["Target"] = df["Close"].shift(-1)

    # Drop initial NaNs from indicators & final NaN from Target
    df = df.dropna()

    return df


# ============================================================
# 📌 CREATE SEQUENCES
# ============================================================
def create_sequences(data: np.ndarray, target: np.ndarray, time_steps: int):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)


# ============================================================
# 📌 TRAINING PIPELINE — stores scalers inside df_processed.attrs
# ============================================================
def train_forecasting_model(df_raw: pd.DataFrame):
    # ---------- Feature engineering ----------
    df = build_features(df_raw)

    # ---------- Train/Valid Split ----------
    split = int(len(df) * 0.85)
    train_df = df.iloc[:split]
    valid_df = df.iloc[split:]

    # ---------- Features used by the model ----------
    # ✅ Explicit list so we can control what the network sees
    feature_cols = [
        "Close", "High", "Low", "Open", "Volume",
        "EMA_20", "EMA_50", "RSI_14", "MACD",
        "Return", "Volatility", "SMA_7"
    ]

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(train_df[feature_cols])
    X_valid_scaled = scaler_x.transform(valid_df[feature_cols])

    # y is NEXT-DAY CLOSE now
    y_train_scaled = scaler_y.fit_transform(train_df[["Target"]])
    y_valid_scaled = scaler_y.transform(valid_df[["Target"]])

    # ---------- Sequence building ----------
    TIME_STEPS = 180

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
    X_valid, y_valid = create_sequences(X_valid_scaled, y_valid_scaled, TIME_STEPS)

    # ---------- Model ----------
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu',
               input_shape=(X_train.shape[1], X_train.shape[2])),
        MaxPooling1D(pool_size=2),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.2),

        Bidirectional(LSTM(64)),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(
        optimizer=AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss='mse'
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=15,
                               restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5,
                                  factor=0.3, min_lr=1e-6)

    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=150,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ---------- Make Predictions on validation ----------
    predictions_scaled = model.predict(X_valid, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    # Actual next-day CLOSE for validation set
    actual = valid_df["Target"].values[TIME_STEPS:]
    actual = actual.reshape(-1, 1)

    # ---------- Save metadata ----------
    df.attrs["scaler_x"] = scaler_x
    df.attrs["scaler_y"] = scaler_y
    df.attrs["feature_cols"] = feature_cols
    df.attrs["time_steps"] = TIME_STEPS

    return model, df, actual, predictions


# ============================================
# 📌 VISUALIZATION (validation)
# ============================================
def plot_forecast(df: pd.DataFrame, actual: np.ndarray, predictions: np.ndarray):
    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=df.index[-len(actual):],
        y=actual.flatten(),
        name="Actual",
        mode="lines",
        line=dict(color="blue")
    ))

    # Predictions
    fig.add_trace(go.Scatter(
        x=df.index[-len(predictions):],
        y=predictions.flatten(),
        name="Predicted",
        mode="lines",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Stock Forecasting — Actual vs Predicted (Next-Day Close)",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    return fig


# ============================================
# 📌 Model Evaluation
# ============================================
def evaluate_model(df, actual, predictions, model, runs=50):
    scaler_x = df.attrs["scaler_x"]
    scaler_y = df.attrs["scaler_y"]
    feature_cols = df.attrs["feature_cols"]
    TIME_STEPS = df.attrs["time_steps"]

    # rebuild validation feature set
    split = int(len(df) * 0.85)
    valid_df = df.iloc[split:]

    X_valid_scaled = scaler_x.transform(valid_df[feature_cols])
    y_dummy = np.zeros((len(valid_df), 1))
    X_valid, _ = create_sequences(X_valid_scaled, y_dummy, TIME_STEPS)
    X_valid = np.array(X_valid)

    actual = actual.flatten()
    predictions = predictions.flatten()

    mse = np.mean((actual - predictions) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - predictions)))
    mape = float(np.mean(np.abs((actual - predictions) / actual)) * 100)

    # Direction of move accuracy
    direction_actual = np.sign(np.diff(actual))
    direction_pred = np.sign(np.diff(predictions))
    direction_accuracy = float(np.mean(direction_actual == direction_pred) * 100)

    # Monte-Carlo dropout for uncertainty
    mc_std_list = []
    for _ in range(runs):
        preds_scaled = model(X_valid, training=True)
        preds = scaler_y.inverse_transform(preds_scaled).flatten()
        mc_std_list.append(np.std(preds))

    avg_uncertainty = float(np.mean(mc_std_list))
    confidence_level = max(0.0, 100.0 - avg_uncertainty)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "direction_accuracy": direction_accuracy,
        "forecast_confidence": confidence_level
    }


# ============================================================
# 📌 FUTURE FORECASTING (60-day auto-regressive)
# ============================================================
def forecast_future(model, df_processed, days=60):
    """
    Auto-regressive multi-step forecast.
    The model predicts NEXT-DAY CLOSE; each prediction is fed back in.
    """

    scaler_x = df_processed.attrs["scaler_x"]
    scaler_y = df_processed.attrs["scaler_y"]
    feature_cols = df_processed.attrs["feature_cols"]
    time_steps = df_processed.attrs["time_steps"]

    # Start from full processed df; we only need the last window of FEATURES
    df_feat = df_processed.copy()
    last_block = df_feat[feature_cols].tail(time_steps).copy()

    # This is our moving feature window (scaled)
    X_window = scaler_x.transform(last_block)
    future_predictions = []

    for _ in range(days):
        X_input = X_window.reshape(1, time_steps, len(feature_cols))

        # 1-step ahead prediction (scaled)
        next_scaled = model.predict(X_input, verbose=0)[0]
        next_close = scaler_y.inverse_transform(next_scaled.reshape(-1, 1))[0, 0]
        future_predictions.append(next_close)

        # ---- Update feature window ----
        # We approximate next feature row using previous row + new Close.
        # This keeps forecast stable without crazy features.
        new_row = last_block.iloc[-1].copy()
        new_row["Close"] = next_close
        new_row["SMA_7"] = (new_row["SMA_7"] * 6 + next_close) / 7.0  # rough update
        new_row["Return"] = (next_close / last_block.iloc[-1]["Close"] - 1.0)
        # Volatility, RSI, MACD, EMA updates could be approximated,
        # but for stability we keep them equal to last row.

        # Shift window and append new row (unscaled)
        last_block = pd.concat([last_block.iloc[1:], new_row.to_frame().T])

        # Rescale entire window for next step
        X_window = scaler_x.transform(last_block[feature_cols])

    return future_predictions


# ============================================================
# 📌 PLOT FUTURE FORECAST
# ============================================================
def plot_future(df_processed, future_preds):
    future_dates = pd.date_range(
        start=df_processed.index[-1] + pd.Timedelta(days=1),
        periods=len(future_preds)
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_processed.index,
        y=df_processed["Close"],
        name="Historical",
        line=dict(color="blue", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        name="Next 60 Days Forecast",
        line=dict(color="red", width=3)
    ))

    fig.update_layout(
        title="60-Day Stock Price Forecast (Next-Day Close)",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    return fig


# ============================================================
# Example (for local testing, not Streamlit)
# ============================================================
#if __name__ == "__main__":
#    ticker = "NCC.NS"
#    df_raw = fetch_data(ticker)
#
#    model, df_processed, actual, predictions = train_forecasting_model(df_raw)
#
#    fig_val = plot_forecast(df_processed, actual, predictions)
#    # fig_val.show()  # uncomment if running locally
#
#    metrics = evaluate_model(df_processed, actual, predictions, model)
#    print("Metrics:", metrics)
#
#    future_60 = forecast_future(model, df_processed, days=60)
#    fig_future = plot_future(df_processed, future_60)
#    # fig_future.show()  # uncomment if running locally
#