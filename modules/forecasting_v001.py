
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

ticker = 'NCC.NS'

def fetch_data(ticker):
    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = [col[0] for col in df.columns]
    return df
    


# ============================================================
# 📌 FEATURE ENGINEERING
# ============================================================
def build_features(df):
    df = df.copy()

    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()

    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(20).std()

    df["Target"] = df["Close"].rolling(7).mean().shift(-1)

    df = df.dropna()
    return df


# ============================================================
# 📌 CREATE SEQUENCES
# ============================================================
def create_sequences(data, target, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)


# ============================================================
# 📌 TRAINING PIPELINE — stores scalers inside df_processed.attrs
# ============================================================
def train_forecasting_model(df_raw):

    # ---------- Feature engineering ----------
    df = build_features(df_raw)

    # ---------- Train/Valid Split ----------
    split = int(len(df) * 0.85)
    train_df = df.iloc[:split]
    valid_df = df.iloc[split:]

    # ---------- Scaling ----------
    feature_cols = df.columns.drop("Target")

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(train_df[feature_cols])
    X_valid_scaled = scaler_x.transform(valid_df[feature_cols])

    y_train_scaled = scaler_y.fit_transform(train_df[["Target"]])
    y_valid_scaled = scaler_y.transform(valid_df[["Target"]])

    # ---------- Sequence building ----------
    TIME_STEPS = 180

    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
    X_valid, y_valid = create_sequences(X_valid_scaled, y_valid_scaled, TIME_STEPS)

    X_train = np.array(X_train)
    X_valid = np.array(X_valid)

    # ---------- Model ----------
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
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

    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=1e-6)

    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=150,
        batch_size=32,
        shuffle=True,
        callbacks=[early_stop, reduce_lr]
    )

    # ---------- Make Predictions ----------
    predictions_scaled = model.predict(X_valid)
    predictions = scaler_y.inverse_transform(predictions_scaled)

    actual = valid_df["Target"].values[TIME_STEPS:]
    actual = actual.reshape(-1, 1)
 
    # ---------- Save metadata ----------
    df.attrs["scaler_x"] = scaler_x
    df.attrs["scaler_y"] = scaler_y
    df.attrs["feature_cols"] = list(feature_cols)
    df.attrs["time_steps"] = TIME_STEPS

    return model, df, actual, predictions


# ============================================
# 📌 VISUALIZATION
# ============================================
def plot_forecast(df, actual, predictions):
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
        title="Stock Forecasting — Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    fig.show()

# ============================================
# Model Evaluation
# ============================================

def evaluate_model(df, actual, predictions, model, runs=50):

    # -----------------------------
    # Load attributes
    # -----------------------------
    scaler_x = df.attrs["scaler_x"]
    scaler_y = df.attrs["scaler_y"]
    feature_cols = df.attrs["feature_cols"]
    TIME_STEPS = df.attrs["time_steps"]

    # -----------------------------
    # Rebuild validation set
    # -----------------------------
    split = int(len(df) * 0.85)
    valid_df = df.iloc[split:]

    X_valid_scaled = scaler_x.transform(valid_df[feature_cols])
    y_dummy = np.zeros((len(valid_df), 1))

    X_valid, _ = create_sequences(X_valid_scaled, y_dummy, TIME_STEPS)
    X_valid = np.array(X_valid)

    # -----------------------------
    # Flatten arrays
    # -----------------------------
    actual = actual.flatten()
    predictions = predictions.flatten()

    # -----------------------------
    # Basic Metrics
    # -----------------------------
    mse = np.mean((actual - predictions) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - predictions)))
    mape = float(np.mean(np.abs((actual - predictions) / actual)) * 100)

    # -----------------------------
    # Direction Accuracy
    # -----------------------------
    direction_actual = np.sign(np.diff(actual))
    direction_pred = np.sign(np.diff(predictions))
    direction_accuracy = float(np.mean(direction_actual == direction_pred) * 100)

    # -----------------------------
    # Confidence Level (Summary Only)
    # -----------------------------
    mc_std_list = []

    for _ in range(runs):
        preds_scaled = model(X_valid, training=True)
        preds = scaler_y.inverse_transform(preds_scaled).flatten()

        mc_std_list.append(np.std(preds))

    avg_uncertainty = float(np.mean(mc_std_list))

    # Convert uncertainty → confidence (inverse relationship)
    confidence_level = max(0, 100 - avg_uncertainty)

    # -----------------------------
    # FINAL CLEAN RETURN (APP READY)
    # -----------------------------
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "direction_accuracy": direction_accuracy,
        "forecast_confidence": confidence_level     # single number %
    }




# ============================================================
# 📌 FUTURE FORECASTING — Only needs df_processed
# ============================================================
def forecast_future(model, df_processed, days=60):
    """
    Auto-regressive 60-day forecast.

    Uses:
      - model
      - df_processed (with attrs: scaler_x, scaler_y, feature_cols, time_steps)
    Returns:
      - list of predicted prices (float)
    """

    scaler_x = df_processed.attrs["scaler_x"]
    scaler_y = df_processed.attrs["scaler_y"]
    feature_cols = df_processed.attrs["feature_cols"]
    time_steps = df_processed.attrs["time_steps"]

    # Start from OHLCV only (no technicals, they’ll be recomputed)
    df_raw_future = df_processed[["Close", "High", "Low", "Open", "Volume"]].copy()

    future_predictions = []

    for _ in range(days):
        # 1) Rebuild technical indicators on full history
        df_feat = build_features(df_raw_future)

        # 2) Take last time_steps window of features
        last_block = df_feat[feature_cols].tail(time_steps)

        X_scaled = scaler_x.transform(last_block)
        X_input = X_scaled.reshape(1, time_steps, len(feature_cols))

        # 3) Predict next (scaled) target and invert
        next_scaled = model.predict(X_input, verbose=0)[0]
        next_unscaled = scaler_y.inverse_transform(next_scaled.reshape(-1, 1))[0, 0]

        # 4) Append new row: copy previous OHLCV, only change Close
        last_row = df_raw_future.iloc[-1].copy()
        next_date = df_raw_future.index[-1] + pd.Timedelta(days=1)

        last_row["Close"] = next_unscaled             # new predicted close
        # High / Low / Open / Volume stay same as last row (no NaNs)

        df_raw_future.loc[next_date] = last_row

        future_predictions.append(next_unscaled)

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
        title="60-Day Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    return fig    # <<<<<< THIS FIXES THE STREAMLIT ERROR




# Example usage

# model, df_processed, actual, predictions = train_forecasting_model(ncc)
# plot_forecast(df_processed, actual, predictions)
# evaluate_model(df_processed, actual, predictions, model, runs=50)
# future_60 = forecast_future(model, df_processed, days=60)
# plot_future(df_processed, future_60)


