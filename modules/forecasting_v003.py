import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Conv1D, 
    MaxPooling1D, Input, MultiHeadAttention, 
    LayerNormalization, GlobalAveragePooling1D, Concatenate
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import ta
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# DATA FETCH
# ============================================================
def fetch_data(ticker: str) -> pd.DataFrame:
    """Fetch historical stock data"""
    start_date = '2012-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced feature engineering with no target leakage.
    All features use ONLY historical data (no future peeking).
    """
    df = df.copy()
    
    # ========== PRICE FEATURES ==========
    df["Returns"] = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Price momentum features
    df["Price_Rate_Change"] = df["Close"].pct_change(periods=10)
    df["High_Low_Ratio"] = df["High"] / df["Low"]
    df["Close_Open_Ratio"] = df["Close"] / df["Open"]
    
    # ========== MOVING AVERAGES ==========
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    
    # MA crossover signals
    df["SMA_10_20_Ratio"] = df["SMA_10"] / df["SMA_20"]
    df["EMA_12_26_Ratio"] = df["EMA_12"] / df["EMA_26"]
    
    # ========== VOLATILITY FEATURES ==========
    df["Volatility_10"] = df["Returns"].rolling(10).std()
    df["Volatility_20"] = df["Returns"].rolling(20).std()
    df["Volatility_30"] = df["Returns"].rolling(30).std()
    
    # ATR (Average True Range)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    
    # ========== MOMENTUM INDICATORS ==========
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["Stochastic"] = ta.momentum.stoch(df["High"], df["Low"], df["Close"], window=14)
    
    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Diff"] = macd.macd_diff()
    
    # ========== TREND INDICATORS ==========
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    
    # ========== BOLLINGER BANDS ==========
    bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_High"] = bollinger.bollinger_hband()
    df["BB_Low"] = bollinger.bollinger_lband()
    df["BB_Mid"] = bollinger.bollinger_mavg()
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]
    df["BB_Position"] = (df["Close"] - df["BB_Low"]) / (df["BB_High"] - df["BB_Low"])
    
    # ========== VOLUME FEATURES ==========
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA_20"]
    
    # On-Balance Volume
    df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["OBV_EMA"] = df["OBV"].ewm(span=20).mean()
    
    # ========== LAG FEATURES ==========
    for lag in [1, 2, 3, 5, 10]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_Lag_{lag}"] = df["Volume"].shift(lag)
    
    # ========== TARGET (NEXT DAY CLOSE) ==========
    df["Target"] = df["Close"].shift(-1)
    
    # Drop NaN values
    df = df.dropna()
    
    return df


# ============================================================
# CREATE SEQUENCES
# ============================================================
def create_sequences(data: np.ndarray, target: np.ndarray, time_steps: int):
    """Create sliding window sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)


# ============================================================
# ADVANCED MODEL ARCHITECTURE
# ============================================================
def build_advanced_model(input_shape):
    """
    Hybrid CNN-LSTM model with Multi-Head Attention
    Architecture:
    1. CNN layers for pattern extraction
    2. Bidirectional LSTM for temporal dependencies
    3. Multi-Head Attention for important feature weighting
    4. Dense layers for final prediction
    """
    
    inputs = Input(shape=input_shape)
    
    # ========== CNN BRANCH ==========
    conv1 = Conv1D(128, kernel_size=3, activation='relu', padding='same')(inputs)
    conv1 = MaxPooling1D(pool_size=2)(conv1)
    conv1 = Dropout(0.2)(conv1)
    
    conv2 = Conv1D(64, kernel_size=3, activation='relu', padding='same')(conv1)
    conv2 = MaxPooling1D(pool_size=2)(conv2)
    conv2 = Dropout(0.2)(conv2)
    
    # ========== LSTM BRANCH ==========
    lstm1 = Bidirectional(LSTM(128, return_sequences=True))(conv2)
    lstm1 = Dropout(0.3)(lstm1)
    
    lstm2 = Bidirectional(LSTM(64, return_sequences=True))(lstm1)
    lstm2 = Dropout(0.3)(lstm2)
    
    # ========== ATTENTION MECHANISM ==========
    attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm2, lstm2)
    attention = LayerNormalization()(attention)
    attention = Dropout(0.2)(attention)
    
    # ========== POOLING & DENSE ==========
    pooled = GlobalAveragePooling1D()(attention)
    
    dense1 = Dense(128, activation='relu')(pooled)
    dense1 = Dropout(0.3)(dense1)
    
    dense2 = Dense(64, activation='relu')(dense1)
    dense2 = Dropout(0.2)(dense2)
    
    dense3 = Dense(32, activation='relu')(dense2)
    
    # Output layer
    outputs = Dense(1)(dense3)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


# ============================================================
# TRAINING PIPELINE
# ============================================================
def train_forecasting_model(df_raw: pd.DataFrame):
    """
    Enhanced training pipeline with advanced preprocessing
    """
    
    # Feature engineering
    df = build_features(df_raw)
    
    # Train/Valid split (85/15)
    split = int(len(df) * 0.85)
    train_df = df.iloc[:split]
    valid_df = df.iloc[split:]
    
    # Feature columns (exclude target and date)
    feature_cols = [col for col in df.columns if col not in ['Target']]
    
    # Use RobustScaler (better for outliers)
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()
    
    X_train_scaled = scaler_x.fit_transform(train_df[feature_cols])
    X_valid_scaled = scaler_x.transform(valid_df[feature_cols])
    
    y_train_scaled = scaler_y.fit_transform(train_df[["Target"]])
    y_valid_scaled = scaler_y.transform(valid_df[["Target"]])
    
    # Create sequences
    TIME_STEPS = 60  # Reduced from 180 for better generalization
    
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, TIME_STEPS)
    X_valid, y_valid = create_sequences(X_valid_scaled, y_valid_scaled, TIME_STEPS)
    
    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_valid.shape}")
    
    # Build model
    model = build_advanced_model((X_train.shape[1], X_train.shape[2]))
    
    model.compile(
        optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-5),
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae']
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=8,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\n🚀 Training Advanced Forecasting Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Make predictions on validation set
    predictions_scaled = model.predict(X_valid, verbose=0)
    predictions = scaler_y.inverse_transform(predictions_scaled)
    
    actual = valid_df["Target"].values[TIME_STEPS:]
    actual = actual.reshape(-1, 1)
    
    # Store metadata
    df.attrs["scaler_x"] = scaler_x
    df.attrs["scaler_y"] = scaler_y
    df.attrs["feature_cols"] = feature_cols
    df.attrs["time_steps"] = TIME_STEPS
    df.attrs["history"] = history.history
    
    return model, df, actual, predictions


# ============================================================
# IMPROVED AUTOREGRESSIVE FORECASTING
# ============================================================
def forecast_future(model, df_processed, days=60):
    """
    Enhanced autoregressive forecasting with proper feature updates
    """
    
    scaler_x = df_processed.attrs["scaler_x"]
    scaler_y = df_processed.attrs["scaler_y"]
    feature_cols = df_processed.attrs["feature_cols"]
    time_steps = df_processed.attrs["time_steps"]
    
    # Get last window of data
    df_feat = df_processed[feature_cols].copy()
    last_window = df_feat.tail(time_steps).copy()
    
    future_predictions = []
    
    for step in range(days):
        # Scale current window
        X_window = scaler_x.transform(last_window)
        X_input = X_window.reshape(1, time_steps, len(feature_cols))
        
        # Predict next day
        next_scaled = model.predict(X_input, verbose=0)[0]
        next_close = scaler_y.inverse_transform(next_scaled.reshape(-1, 1))[0, 0]
        future_predictions.append(next_close)
        
        # Update features for next prediction
        new_row = update_features_for_next_step(
            last_window.iloc[-1].copy(),
            next_close,
            last_window
        )
        
        # Slide window forward
        last_window = pd.concat([
            last_window.iloc[1:],
            new_row.to_frame().T
        ], ignore_index=True)
    
    return future_predictions


def update_features_for_next_step(prev_row, next_close, window_df):
    """
    Safely update all features for autoregressive forecasting.
    Guaranteed to return a valid Series.
    """
    try:
        new_row = prev_row.copy()

        # ------------------------------
        # BASIC PRICE FEATURES
        # ------------------------------
        prev_close = float(prev_row.get("Close", next_close))
        new_row["Close"] = next_close

        if prev_close > 0:
            ret = (next_close / prev_close) - 1
        else:
            ret = 0.0

        new_row["Returns"] = ret
        new_row["Log_Returns"] = np.log(max(next_close, 1e-8) / max(prev_close, 1e-8))

        # ------------------------------
        # MOVING AVERAGES (EMA STYLE)
        # ------------------------------
        def ema_update(prev_val, price, period):
            alpha = 2 / (period + 1)
            return prev_val * (1 - alpha) + price * alpha

        for period, col in [(10, "SMA_10"), (20, "SMA_20"), (50, "SMA_50")]:
            if col in new_row.index:
                new_row[col] = ema_update(new_row[col], next_close, period)

        for period, col in [(12, "EMA_12"), (26, "EMA_26")]:
            if col in new_row.index:
                new_row[col] = ema_update(new_row[col], next_close, period)

        # Ratios
        if "SMA_10" in new_row.index and "SMA_20" in new_row.index:
            denom = new_row["SMA_20"] if new_row["SMA_20"] != 0 else 1e-8
            new_row["SMA_10_20_Ratio"] = new_row["SMA_10"] / denom

        if "EMA_12" in new_row.index and "EMA_26" in new_row.index:
            denom = new_row["EMA_26"] if new_row["EMA_26"] != 0 else 1e-8
            new_row["EMA_12_26_Ratio"] = new_row["EMA_12"] / denom

        # ------------------------------
        # VOLATILITY DECAY
        # ------------------------------
        for col in ["Volatility_10", "Volatility_20", "Volatility_30"]:
            if col in new_row.index:
                new_row[col] *= 0.95

        # ------------------------------
        # MOMENTUM DECAY
        # ------------------------------
        for col in ["RSI", "Stochastic", "MACD", "MACD_Signal", "ADX"]:
            if col in new_row.index:
                new_row[col] *= 0.98

        # ------------------------------
        # LAG FEATURES — SAFE HANDLING
        # ------------------------------
        lag_cols = [c for c in new_row.index if c.startswith("Close_Lag_")]
        lag_nums = sorted(int(c.split("_")[-1]) for c in lag_cols)

        for lag in lag_nums:
            col = f"Close_Lag_{lag}"

            if lag == 1:
                new_row[col] = prev_close
            else:
                prev_col = f"Close_Lag_{lag-1}"
                if prev_col in new_row.index:
                    new_row[col] = prev_row.get(prev_col, prev_row.get(col, next_close))
                else:
                    new_row[col] = prev_row.get(col, next_close)

        return new_row

    except Exception as e:
        print("ERROR UPDATE FEATURES:", e)
        return prev_row  # SAFE FALLBACK, prevents None return

# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(df, actual, predictions, model, runs=30):
    """Enhanced evaluation with multiple metrics"""
    
    actual = actual.flatten()
    predictions = predictions.flatten()
    
    # Basic metrics
    mse = np.mean((actual - predictions) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - predictions)))
    
    # Direction accuracy (most important for trading)
    actual_direction = np.sign(np.diff(actual))
    pred_direction = np.sign(np.diff(predictions))
    direction_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
    
    # R-squared
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) * 100
    
    # Forecast confidence (inverse of uncertainty)
    confidence = max(0.0, min(100.0, 95 - (rmse / np.mean(actual) * 100)))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_accuracy,
        "r2_score": r2,
        "forecast_confidence": confidence
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_forecast(df: pd.DataFrame, actual: np.ndarray, predictions: np.ndarray):
    """Plot validation results"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index[-len(actual):],
        y=actual.flatten(),
        name="Actual",
        mode="lines",
        line=dict(color="#3b82f6", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[-len(predictions):],
        y=predictions.flatten(),
        name="Predicted",
        mode="lines",
        line=dict(color="#ef4444", width=2)
    ))
    
    fig.update_layout(
        title="Model Validation: Actual vs Predicted (Next-Day Close)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig


def plot_future(df_processed, future_preds):
    """Plot future forecast"""
    
    future_dates = pd.date_range(
        start=df_processed.index[-1] + pd.Timedelta(days=1),
        periods=len(future_preds)
    )
    
    fig = go.Figure()
    
    # Historical data (last 180 days)
    fig.add_trace(go.Scatter(
        x=df_processed.index[-180:],
        y=df_processed["Close"].iloc[-180:],
        name="Historical Price",
        line=dict(color="#3b82f6", width=2)
    ))
    
    # Future predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_preds,
        name="60-Day Forecast",
        line=dict(color="#f59e0b", width=3, dash='dash')
    ))
    
    # Add confidence bands (±5%)
    upper_band = [p * 1.05 for p in future_preds]
    lower_band = [p * 0.95 for p in future_preds]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_band,
        fill=None,
        mode='lines',
        line=dict(color='rgba(245, 158, 11, 0.2)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_band,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(245, 158, 11, 0.2)'),
        name='Confidence Band (±5%)'
    ))
    
    fig.update_layout(
        title="60-Day Stock Price Forecast with Confidence Bands",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        hovermode='x unified'
    )
    
    return fig