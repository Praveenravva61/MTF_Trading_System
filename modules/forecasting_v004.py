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
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Dropout,
    Bidirectional, LSTM,
    MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
# 1. MODEL: DIRECT 60-DAY OUTPUT (NO AUTOREGRESSION)
# ============================================================

def build_advanced_model(input_shape, horizon=60):
    """
    Hybrid CNN-BiLSTM + Multi-Head Attention
    Direct multi-step head: predicts `horizon` future prices at once.
    """
    inputs = Input(shape=input_shape)

    # ----- CNN branch -----
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # ----- BiLSTM -----
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.3)(x)

    # ----- Multi-head Attention -----
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    attn = LayerNormalization()(attn)
    attn = Dropout(0.2)(attn)

    # ----- Pooling + Dense -----
    pooled = GlobalAveragePooling1D()(attn)

    d = Dense(128, activation='relu')(pooled)
    d = Dropout(0.3)(d)

    d = Dense(64, activation='relu')(d)
    d = Dropout(0.2)(d)

    d = Dense(32, activation='relu')(d)

    # ----- Direct multi-step head -----
    outputs = Dense(horizon)(d)   # <--- 60 future closes

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================
# 2. SEQUENCE CREATION: MULTI-STEP TARGET
# ============================================================

def create_sequences_multi_step(X, y, time_steps, horizon):
    """
    X: (N, n_features)
    y: (N, 1)   # target close (scaled)
    Returns:
        Xs: (num_samples, time_steps, n_features)
        ys: (num_samples, horizon)   # each row = next 60 closes
    """
    Xs, ys = [], []
    N = len(X)

    for i in range(N - time_steps - horizon + 1):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps : i + time_steps + horizon].flatten())

    return np.array(Xs), np.array(ys)


# ============================================================
# 3. TRAINING PIPELINE (DIRECT 60-DAY FORECAST)
# ============================================================
def train_forecasting_model(df_raw: pd.DataFrame,
                            time_steps: int = 60,
                            horizon: int = 60):
    """
    Direct multi-step training:
    Input  : last `time_steps` days
    Output : next `horizon` days (60) of Target/Close.
    """

    # 1) Feature engineering (your existing function)
    df = build_features(df_raw)   # must create 'Target' column

    # 2) Train/Valid split
    split = int(len(df) * 0.85)
    train_df = df.iloc[:split]
    valid_df = df.iloc[split:]

    # 3) Feature columns (everything except Target)
    feature_cols = [c for c in df.columns if c != 'Target']

    scaler_x = RobustScaler()
    scaler_y = RobustScaler()

    X_train_scaled = scaler_x.fit_transform(train_df[feature_cols])
    X_valid_scaled = scaler_x.transform(valid_df[feature_cols])

    y_train_scaled = scaler_y.fit_transform(train_df[["Target"]])
    y_valid_scaled = scaler_y.transform(valid_df[["Target"]])

    # 4) Create multi-step sequences
    X_train, y_train = create_sequences_multi_step(
        X_train_scaled, y_train_scaled, time_steps, horizon
    )
    X_valid, y_valid = create_sequences_multi_step(
        X_valid_scaled, y_valid_scaled, time_steps, horizon
    )

    print(f"Training shape X: {X_train.shape}, y: {y_train.shape}")
    print(f"Validation shape X: {X_valid.shape}, y: {y_valid.shape}")

    # 5) Build & compile model
    model = build_advanced_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        horizon=horizon
    )

    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="huber",   # robust to outliers
        metrics=["mae"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        patience=8,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    )

    print("\n🚀 Training Direct 60-Day Forecasting Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # 6) Validation predictions (for plots & metrics)
    preds_scaled = model.predict(X_valid, verbose=0)     # (samples, horizon)
    preds_full = scaler_y.inverse_transform(preds_scaled)  # unscaled

    actual_full = scaler_y.inverse_transform(y_valid)      # (samples, horizon)

    # For "Model Validation: Actual vs Predicted (Next-Day Close)"
    # compare only the first horizon step (t+1)
    pred_next = preds_full[:, 0].reshape(-1, 1)
    actual_next = actual_full[:, 0].reshape(-1, 1)

    residuals = actual_next - pred_next
    residual_std = float(np.std(residuals))

    # Store metadata in df for later forecasting
    df.attrs["scaler_x"] = scaler_x
    df.attrs["scaler_y"] = scaler_y
    df.attrs["feature_cols"] = feature_cols
    df.attrs["time_steps"] = time_steps
    df.attrs["horizon"] = horizon
    df.attrs["history"] = history.history

    return model, df, actual_next, pred_next, residual_std


# ============================================================
# 4. DIRECT 60-DAY FORECAST (NO RECURSION)
# ============================================================

def forecast_future(model,
                    df_processed: pd.DataFrame,
                    days: int = 60):
    """
    Direct multi-step forecast.
    Uses ONLY last `time_steps` of ORIGINAL data, no predicted feedback.
    """
    scaler_x = df_processed.attrs["scaler_x"]
    scaler_y = df_processed.attrs["scaler_y"]
    feature_cols = df_processed.attrs["feature_cols"]
    time_steps = df_processed.attrs["time_steps"]
    horizon = df_processed.attrs.get("horizon", 60)

    days = min(days, horizon)

    # Use last `time_steps` rows of REAL data
    df_feat = df_processed[feature_cols].tail(time_steps).values

    X_window = scaler_x.transform(df_feat)
    X_input = X_window.reshape(1, time_steps, len(feature_cols))

    y_scaled = model.predict(X_input, verbose=0)[0]          # (horizon,)
    y_unscaled = scaler_y.inverse_transform(y_scaled.reshape(-1, 1))[:, 0]

    return y_unscaled[:days]


# ============================================================
# 5. OPTIONAL: FORECAST WITH CONFIDENCE BANDS
# ============================================================

def forecast_future_with_ci(model,
                            df_processed: pd.DataFrame,
                            days: int = 60,
                            residual_std: float = None,
                            quantiles=(0.1, 0.5, 0.9),
                            n_samples: int = 500):
    """
    Wraps `forecast_future` and adds simple Gaussian confidence bands
    using residual_std from validation.
    """
    path = forecast_future(model, df_processed, days=days)
    days = len(path)

    if residual_std is None or residual_std <= 0:
        return {
            "path": path,
            "forecast_df": pd.DataFrame({"Close_Pred": path})
        }

    # Sample noise around direct path
    noise = np.random.normal(0.0, residual_std, size=(n_samples, days))
    samples = path[None, :] + noise

    qs = sorted(list(quantiles))
    q_vals = {q: np.quantile(samples, q, axis=0) for q in qs}

    # Build date index (business days)
    if isinstance(df_processed.index, pd.DatetimeIndex):
        last_date = df_processed.index[-1]
    else:
        last_date = pd.to_datetime(df_processed.iloc[-1].name)

    future_dates = pd.bdate_range(start=last_date, periods=days + 1)[1:]

    forecast_df = pd.DataFrame(index=future_dates)
    forecast_df["Close_Pred"] = path
    for q in qs:
        forecast_df[f"q_{int(q * 100):02d}"] = q_vals[q]

    return {
        "path": path,
        "samples": samples,
        "forecast_df": forecast_df,
    }

# ============================================================
# EVALUATION
# ============================================================
def evaluate_model(df, actual_next, pred_next, model=None, runs=30):
    """
    Evaluation metrics for next-day validation predictions
    Works with direct multi-step forecasting pipeline.
    """

    # Convert to 1D arrays
    actual = actual_next.flatten()
    predictions = pred_next.flatten()

    # ----- BASIC METRICS -----
    mse = np.mean((actual - predictions) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - predictions)))

    # ----- DIRECTION ACCURACY -----
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predictions))
    direction_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

    # ----- R-SQUARED -----
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) * 100

    # ----- CONFIDENCE SCORE (simple heuristic) -----
    # smaller rmse → higher confidence
    confidence = max(0.0, min(100.0, 95 - (rmse / np.mean(actual) * 100)))

    return {
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_accuracy,
        "r2_score": r2,
        "forecast_confidence": confidence,
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_forecast(df: pd.DataFrame, actual_next: np.ndarray, pred_next: np.ndarray):
    """Plot validation results for next-day forecast."""
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index[-len(actual_next):],
        y=actual_next.flatten(),
        name="Actual (Next-Day)",
        mode="lines",
        line=dict(color="#3b82f6", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index[-len(pred_next):],
        y=pred_next.flatten(),
        name="Predicted (Next-Day)",
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


def plot_future(df_processed,forecast_df):
    
    """
    Plot 60-day forecast with confidence intervals.
    forecast_df must contain:
        - Close_Pred
        - q_10, q_50, q_90  (optional)
    """

    future_dates = forecast_df.index
    preds = forecast_df["Close_Pred"].values

    fig = go.Figure()

    # ---- Historical Prices (last 180 days) ----
    fig.add_trace(go.Scatter(
        x=df_processed.index[-180:],
        y=df_processed["Close"].iloc[-180:],
        name="Historical Price",
        line=dict(color="#3b82f6", width=2)
    ))

    # ---- Forecast Line ----
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=preds,
        name="60-Day Forecast",
        line=dict(color="#f59e0b", width=3, dash='dash')
    ))

    # ---- Confidence Bands if present ----
    q_cols = [c for c in forecast_df.columns if c.startswith("q_")]
    if len(q_cols) >= 2:
        lower = forecast_df[q_cols[0]]
        upper = forecast_df[q_cols[-1]]

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=upper,
            fill=None,
            mode='lines',
            line=dict(color='rgba(245, 158, 11, 0.2)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=future_dates,
            y=lower,
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(245, 158, 11, 0.2)'),
            name='Confidence Band'
        ))

    fig.update_layout(
        title="60-Day Stock Price Forecast with Confidence Bands",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        template="plotly_dark",
        hovermode='x unified'
    )

    return fig
