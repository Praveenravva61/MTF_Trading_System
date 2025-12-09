import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, Conv1D, 
    MaxPooling1D, Input, MultiHeadAttention, 
    LayerNormalization, GlobalAveragePooling1D, Concatenate, Add, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
    # Using auto_adjust=True to handle splits/dividends better
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df


# ============================================================
# ADVANCED FEATURE ENGINEERING (STATIONARY FOCUS)
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features are engineered to be STATIONARY (ratios/percentages).
    Raw prices are dropped from the feature set to prevent scaling bias.
    """
    df = df.copy()
    
    # 1. Log Returns (The core trend feature)
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # 2. Price Dynamics (Ratios) - Normalized
    df["High_Low_Ratio"] = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_Ratio"] = (df["Close"] - df["Open"]) / df["Close"]
    
    # 3. Moving Average Ratios (Distance from MA)
    # Instead of raw SMA, use Close/SMA - 1
    for win in [10, 20, 50]:
        sma = df["Close"].rolling(win).mean()
        df[f"Dist_SMA_{win}"] = (df["Close"] / sma) - 1
    
    df["EMA_12"] = df["Close"].ewm(span=12).mean()
    df["EMA_26"] = df["Close"].ewm(span=26).mean()
    df["MACD_Line"] = (df["EMA_12"] - df["EMA_26"]) / df["Close"] # Normalize by price
    
    # 4. Volatility & Momentum
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14) / 100.0  # Scale 0-1
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    df["ATR_Pct"] = df["ATR"] / df["Close"]  # Relative volatility
    
    # 5. Bollinger Bands (Position)
    bollinger = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_Width"] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / bollinger.bollinger_mavg()
    df["BB_Pos"] = (df["Close"] - bollinger.bollinger_lband()) / (bollinger.bollinger_hband() - bollinger.bollinger_lband())
    
    # 6. Volume
    df["Volume_Log"] = np.log(df["Volume"] + 1)
    df["Vol_SMA_20"] = df["Volume_Log"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume_Log"] / df["Vol_SMA_20"]
    
    # Drop NaN
    df = df.dropna()
    
    # Identify feature columns (Exclude raw price columns)
    # We keep 'Close' only for reconstruction, not as a model input
    feature_cols = [
        "Log_Returns", "High_Low_Ratio", "Close_Open_Ratio",
        "Dist_SMA_10", "Dist_SMA_20", "Dist_SMA_50", "MACD_Line",
        "RSI", "ATR_Pct", "BB_Width", "BB_Pos", "Vol_Ratio"
    ]
    
    # Store feature names in attributes
    df.attrs["feature_cols"] = feature_cols
    
    return df


# ============================================================
# 1. MODEL: RESIDUAL LSTM-CNN
# ============================================================
def build_advanced_model(input_shape, horizon=60):
    """
    Simplified but robust architecture.
    Uses Residual connections for better gradient flow.
    """
    inputs = Input(shape=input_shape)

    # 1. CNN Block for local feature extraction
    x = Conv1D(64, kernel_size=3, padding='same', activation='linear')(inputs)
    x = LayerNormalization()(x)
    x = Activation('gelu')(x)
    x = Dropout(0.2)(x)
    
    # 2. Bi-LSTM Block
    # Return sequences=True to keep time dimension for Attention
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm_out = LayerNormalization()(lstm_out)
    
    # 3. Attention Mechanism
    # Queries = lstm_out, Keys/Values = lstm_out
    attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_out, lstm_out)
    attn_out = Dropout(0.2)(attn_out)
    x = Add()([lstm_out, attn_out])  # Residual connection
    x = LayerNormalization()(x)
    
    # 4. Global Pooling
    x = GlobalAveragePooling1D()(x)
    
    # 5. Dense Head
    x = Dense(64, activation='gelu')(x)
    x = Dropout(0.2)(x)
    
    # Output: Predicted Cumulative Log Returns for next 'horizon' steps
    outputs = Dense(horizon)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================
# 2. SEQUENCE CREATION: PREDICTING RELATIVE RETURNS
# ============================================================
def create_sequences_relative(df, feature_cols, time_steps, horizon):
    """
    Creates inputs (X) and targets (y).
    X: Window of stationary features.
    y: Cumulative Log Returns relative to the LAST price in the input window.
    """
    data = df[feature_cols].values
    close_prices = df["Close"].values
    
    X, y = [], []
    
    # Need sufficient data for window + horizon
    for i in range(len(df) - time_steps - horizon + 1):
        # Input: features from t to t+time_steps
        X.append(data[i : i + time_steps])
        
        # Target construction:
        # Get actual prices for the forecast horizon
        current_price = close_prices[i + time_steps - 1] # Last known price in input
        future_prices = close_prices[i + time_steps : i + time_steps + horizon]
        
        # Calculate Cumulative Log Return: log(P_future / P_current)
        # This handles scale perfectly (unbounded prices become bounded ratios)
        target_returns = np.log(future_prices / current_price)
        y.append(target_returns)
        
    return np.array(X), np.array(y)


# ============================================================
# 3. TRAINING PIPELINE
# ============================================================
def train_forecasting_model(df_raw: pd.DataFrame,
                            time_steps: int = 60,
                            horizon: int = 60):
    
    # 1. Feature Engineering
    df = build_features(df_raw)
    feature_cols = df.attrs["feature_cols"]
    
    # 2. Train/Valid Split
    split_idx = int(len(df) * 0.90) # 90% Train, 10% Validation
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]
    
    # 3. Scaling (Fit on Train, Apply to Valid)
    # Note: We scale FEATURES, but Targets are Log Returns (already small scale ~ +/- 0.5)
    scaler_x = RobustScaler()
    
    train_feats = scaler_x.fit_transform(train_df[feature_cols])
    valid_feats = scaler_x.transform(valid_df[feature_cols])
    
    # Helper to reconstruct dataframe with scaled values for sequence generation
    df_train_scaled = pd.DataFrame(train_feats, columns=feature_cols, index=train_df.index)
    df_train_scaled["Close"] = train_df["Close"].values # Keep raw close for target gen
    
    df_valid_scaled = pd.DataFrame(valid_feats, columns=feature_cols, index=valid_df.index)
    df_valid_scaled["Close"] = valid_df["Close"].values
    
    # 4. Create Sequences
    X_train, y_train = create_sequences_relative(df_train_scaled, feature_cols, time_steps, horizon)
    X_valid, y_valid = create_sequences_relative(df_valid_scaled, feature_cols, time_steps, horizon)
    
    print(f"Training shape: {X_train.shape}, {y_train.shape}")

    # 5. Build Model
    model = build_advanced_model((X_train.shape[1], X_train.shape[2]), horizon)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="mse",  # MSE is good for log returns
        metrics=["mae"]
    )
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    print("\n🚀 Training Direct 60-Day Return Model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # 6. Generate Validation Predictions (Price Reconstruction)
    # We predict Log Returns, then convert back to Price
    pred_returns = model.predict(X_valid, verbose=0)
    
    # Reconstruct Prices for the "Next Day" validation check
    # We need the reference price (last price in input window)
    # X_valid indices start from 'split_idx' + 'time_steps' roughly
    # Correct way: use the Close price stored in df_valid_scaled at appropriate index
    
    # The 'last input price' for validation sample 'i' corresponds to valid_df index (i + time_steps - 1)
    valid_close_prices = df_valid_scaled["Close"].values
    ref_prices = valid_close_prices[time_steps - 1 : -horizon] 
    
    # If lengths slightly mismatch due to slicing, adjust:
    min_len = min(len(ref_prices), len(pred_returns))
    ref_prices = ref_prices[:min_len]
    pred_returns = pred_returns[:min_len]
    actual_returns = y_valid[:min_len]
    
    # Reconstruct: Price_t+k = Price_t * exp(Cumulative_Log_Return)
    # For validation plot, we usually just look at t+1 (next day)
    pred_next_day_log = pred_returns[:, 0]
    actual_next_day_log = actual_returns[:, 0]
    
    pred_next_price = ref_prices * np.exp(pred_next_day_log)
    actual_next_price = ref_prices * np.exp(actual_next_day_log)
    
    residuals = actual_next_price - pred_next_price
    residual_std = float(np.std(residuals))
    
    # Store metadata
    df.attrs["scaler_x"] = scaler_x
    df.attrs["feature_cols"] = feature_cols
    df.attrs["time_steps"] = time_steps
    df.attrs["horizon"] = horizon
    
    return model, df, actual_next_price.reshape(-1,1), pred_next_price.reshape(-1,1), residual_std


# ============================================================
# 4. FORECASTING FUTURE
# ============================================================
def forecast_future(model, df_processed: pd.DataFrame, days: int = 60):
    """
    Predicts future returns and reconstructs price path.
    """
    scaler_x = df_processed.attrs["scaler_x"]
    feature_cols = df_processed.attrs["feature_cols"]
    time_steps = df_processed.attrs["time_steps"]
    
    # 1. Prepare Input (Last window)
    last_window_df = df_processed.iloc[-time_steps:]
    last_close = last_window_df["Close"].iloc[-1]
    
    # Scale features
    input_feats = scaler_x.transform(last_window_df[feature_cols])
    X_input = input_feats.reshape(1, time_steps, len(feature_cols))
    
    # 2. Predict (Returns)
    pred_log_returns = model.predict(X_input, verbose=0)[0] # Shape (60,)
    
    # 3. Reconstruct Prices
    # Price_future = Last_Price * exp(Predicted_Log_Return)
    future_prices = last_close * np.exp(pred_log_returns)
    
    return future_prices[:days]


# ============================================================
# 5. FORECAST WITH CI (Wrapper)
# ============================================================
def forecast_future_with_ci(model, df_processed, days=60, residual_std=None, quantiles=(0.1, 0.5, 0.9), n_samples=500):
    path = forecast_future(model, df_processed, days=days)
    
    if residual_std is None: residual_std = path[0] * 0.02 # Fallback 2%
    
    # Generate simpler bands based on residual growth
    # Uncertainty grows with sqrt(time)
    time_idxs = np.arange(1, len(path)+1)
    std_devs = residual_std * np.sqrt(time_idxs)
    
    forecast_df = pd.DataFrame(index=pd.date_range(start=df_processed.index[-1], periods=len(path)+1, freq='B')[1:])
    forecast_df["Close_Pred"] = path
    
    # Simple Gaussian bands
    forecast_df["q_10"] = path - 1.645 * std_devs
    forecast_df["q_50"] = path
    forecast_df["q_90"] = path + 1.645 * std_devs
    
    # Clip lower to be realistic
    forecast_df[forecast_df < 0] = 0
    
    return {
        "path": path,
        "forecast_df": forecast_df
    }

# ============================================================
# EVALUATION & PLOTTING (UNCHANGED)
# ============================================================
def evaluate_model(df, actual_next, pred_next, model=None, runs=30):
    actual = actual_next.flatten()
    predictions = pred_next.flatten()
    mse = np.mean((actual - predictions) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(actual - predictions)))
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predictions))
    direction_accuracy = float(np.mean(actual_dir == pred_dir) * 100)
    ss_res = np.sum((actual - predictions) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) * 100
    confidence = max(0.0, min(100.0, 95 - (rmse / np.mean(actual) * 100)))
    return {
        "rmse": rmse, "mae": mae, "direction_accuracy": direction_accuracy,
        "r2_score": r2, "forecast_confidence": confidence
    }

def plot_forecast(df, actual_next, pred_next):
    fig = go.Figure()
    # Align dates
    dates = df.index[-len(actual_next):]
    fig.add_trace(go.Scatter(x=dates, y=actual_next.flatten(), name="Actual", line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=dates, y=pred_next.flatten(), name="Predicted", line=dict(color="#ef4444")))
    fig.update_layout(title="Validation: Next Day Prediction", template="plotly_dark")
    return fig

def plot_future(df_processed, forecast_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_processed.index[-100:], y=df_processed["Close"].iloc[-100:], name="History", line=dict(color="#3b82f6")))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Close_Pred"], name="Forecast", line=dict(color="#f59e0b", dash='dash')))
    if "q_90" in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["q_90"], line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["q_10"], fill='tonexty', line=dict(width=0), fillcolor='rgba(245, 158, 11, 0.2)', name='Confidence'))
    fig.update_layout(title="Future Forecast", template="plotly_dark")
    return fig