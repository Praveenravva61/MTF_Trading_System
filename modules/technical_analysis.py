"""Technical Analysis Module"""
import numpy as np
import pandas as pd


def validate_df(df):
    """Validate and prepare DataFrame for technical analysis."""
    expected = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.copy()
    try:
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
    except:
        df.index = pd.to_datetime(df.index, errors="coerce")

    if df[expected].isnull().any().any():
        df[expected] = df[expected].ffill().bfill()

    return df


def sma(series, period):
    """Simple Moving Average."""
    return series.rolling(period, min_periods=1).mean()


def ema(series, period):
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    """Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    """MACD Indicator."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series, period=20, num_std=2):
    """Bollinger Bands."""
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def directional_indicators(df, period=14):
    """ADX and Directional Indicators."""
    high, low, close = df['High'], df['Low'], df['Close']

    up_move = high.diff()
    down_move = low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_series)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_series)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace(0, np.nan) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return plus_di.fillna(0), minus_di.fillna(0), adx.fillna(0)


def stochastic_oscillator(df, k_period=14, d_period=3):
    """Stochastic Oscillator."""
    low_min = df['Low'].rolling(k_period).min()
    high_max = df['High'].rolling(k_period).max()
    k = 100 * (df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)
    d = k.rolling(d_period).mean()
    return k.fillna(50), d.fillna(50)


def obv(df):
    """On-Balance Volume."""
    obv_vals = np.zeros(len(df))
    for i in range(1, len(df)):
        if df["Close"].iat[i] > df["Close"].iat[i-1]:
            obv_vals[i] = obv_vals[i-1] + df["Volume"].iat[i]
        elif df["Close"].iat[i] < df["Close"].iat[i-1]:
            obv_vals[i] = obv_vals[i-1] - df["Volume"].iat[i]
        else:
            obv_vals[i] = obv_vals[i-1]
    return pd.Series(obv_vals, index=df.index)


def signal_sma_cross(df):
    """SMA Crossover Signal."""
    s50 = sma(df['Close'], 50)
    s200 = sma(df['Close'], 200)
    return pd.Series(np.where(s50 > s200, 1, np.where(s50 < s200, -1, 0)), index=df.index)


def signal_ema_cross(df):
    """EMA Crossover Signal."""
    e12, e26 = ema(df['Close'], 12), ema(df['Close'], 26)
    return pd.Series(np.where(e12 > e26, 1, np.where(e12 < e26, -1, 0)), index=df.index)


def signal_macd(df):
    """MACD Signal."""
    macd_line, signal_line, _ = macd(df['Close'])
    sig = np.where(macd_line > signal_line, 1, np.where(macd_line < signal_line, -1, 0))
    return pd.Series(sig, index=df.index), macd_line, signal_line


def signal_rsi(df):
    """RSI Signal."""
    r = rsi(df['Close'])
    sig = np.where(r < 30, 1, np.where(r > 70, -1, 0))
    return pd.Series(sig, index=df.index), r


def signal_bollinger(df):
    """Bollinger Bands Signal."""
    upper, mid, lower = bollinger_bands(df['Close'])
    sig = np.where(df['Close'] < lower, 1, np.where(df['Close'] > upper, -1, 0))
    return pd.Series(sig, index=df.index), upper, mid, lower


def signal_stoch(df):
    """Stochastic Signal."""
    k, d = stochastic_oscillator(df)
    sig = np.where((k < 20) & (k > d), 1, np.where((k > 80) & (k < d), -1, 0))
    return pd.Series(sig, index=df.index), k, d


def signal_obv(df):
    """OBV Signal."""
    obv_series = obv(df)
    sig = np.where((obv_series.diff(5) > 0) & (df['Close'].diff(5) > 0), 1,
                   np.where((obv_series.diff(5) < 0) & (df['Close'].diff(5) < 0), -1, 0))
    return pd.Series(sig, index=df.index), obv_series


def signal_adx(df):
    """ADX Signal."""
    plus_di, minus_di, adx = directional_indicators(df)
    sig = np.where((adx > 25) & (plus_di > minus_di), 1,
                   np.where((adx > 25) & (plus_di < minus_di), -1, 0))
    return pd.Series(sig, index=df.index), plus_di, minus_di, adx


def aggregate_signals(signals, weights=None):
    """Aggregate multiple signals into final signal."""
    if weights is None:
        weights = {
            'sma': 2, 'ema': 2, 'macd': 3,
            'rsi': 1.5, 'boll': 1.5, 'stoch': 1,
            'obv': 1, 'adx': 2
        }

    w = np.array([weights[c] for c in signals.columns])
    score = (signals.values * w).sum(axis=1) / w.sum()
    signal = np.where(score > 0.15, "BUY", np.where(score < -0.15, "SELL", "HOLD"))
    return score, signal


def momentum_label(macd_line, macd_signal, rsi_val, adx_val):
    """Determine momentum label."""
    macd_slope = (macd_line - macd_signal).iloc[-3:].mean()

    if macd_slope > 0 and rsi_val > 50 and adx_val > 25:
        return "Bullish"
    if macd_slope < 0 and rsi_val < 50 and adx_val > 25:
        return "Bearish"
    return "Neutral"


def apply_technical_analysis(df):
    """Main technical analysis function."""
    df = validate_df(df).copy()

    sma_sig = signal_sma_cross(df)
    ema_sig = signal_ema_cross(df)

    macd_sig, macd_line, macd_signal = signal_macd(df)
    rsi_sig, rsi_val = signal_rsi(df)
    boll_sig, upper, mid, lower = signal_bollinger(df)
    stoch_sig, k, d = signal_stoch(df)
    obv_sig, obv_series = signal_obv(df)
    adx_sig, plus_di, minus_di, adx_val = signal_adx(df)

    signals = pd.DataFrame({
        'sma': sma_sig,
        'ema': ema_sig,
        'macd': macd_sig,
        'rsi': rsi_sig,
        'boll': boll_sig,
        'stoch': stoch_sig,
        'obv': obv_sig,
        'adx': adx_sig
    }, index=df.index)

    score, fsig = aggregate_signals(signals)
    df["TA_SCORE"] = score
    df["TA_SIGNAL"] = fsig

    momentum = momentum_label(macd_line, macd_signal, rsi_val.iloc[-1], adx_val.iloc[-1])

    rationale = "; ".join(
        f"{col.upper()}: {'bullish' if val==1 else 'bearish' if val==-1 else 'neutral'}"
        for col, val in signals.iloc[-1].items()
    )

    summary = {
        "date": df.index[-1],
        "final_signal": fsig[-1],
        "score": float(score[-1]),
        "momentum": momentum,
        "rationale": rationale,
        "indicators": {
            "RSI": float(rsi_val.iloc[-1]),
            "MACD": float(macd_line.iloc[-1]),
            "MACD_SIGNAL": float(macd_signal.iloc[-1]),
            "ADX": float(adx_val.iloc[-1]),
            "SMA50": float(sma(df['Close'], 50).iloc[-1]),
            "SMA200": float(sma(df['Close'], 200).iloc[-1]),
            "STOCH_K": float(k.iloc[-1]),
            "STOCH_D": float(d.iloc[-1]),
            "OBV": float(obv_series.iloc[-1])
        }
    }

    return df, signals, summary