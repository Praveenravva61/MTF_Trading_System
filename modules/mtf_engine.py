"""Multi-Timeframe Trend Aggregation Engine"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def ensure_datetime_index(df: pd.DataFrame, datetime_col: str = "Datetime") -> pd.DataFrame:
    """Ensure DataFrame has datetime index."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if datetime_col in df.columns:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df = df.set_index(datetime_col)
        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                raise ValueError("df must have a DatetimeIndex or a 'Datetime' column parseable to datetime.")
    df = df.sort_index()
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe."""
    return df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna(how='any')


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def macd(df_close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD Indicator."""
    ema_fast = ema(df_close, fast)
    ema_slow = ema(df_close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def find_pivots(series: pd.Series, left: int = 3, right: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Identify local pivot highs and lows."""
    s = series
    n = len(s)
    ph = np.zeros(n, dtype=bool)
    pl = np.zeros(n, dtype=bool)
    arr = s.values
    for i in range(left, n - right):
        window = arr[i-left:i+right+1]
        center = arr[i]
        if center == window.max() and (window != center).any():
            ph[i] = True
        if center == window.min() and (window != center).any():
            pl[i] = True
    return pd.Series(ph, index=series.index), pd.Series(pl, index=series.index)


def hh_hl_structure_from_pivots(df: pd.DataFrame, lookback_pivots: int = 3) -> str:
    """Determine basic structural bias."""
    highs = df['High']
    lows = df['Low']
    ph, pl = find_pivots(highs, left=3, right=3)
    
    pivot_highs_idx = ph[ph].index
    pivot_lows_idx = pl[pl].index

    last_highs = list(highs.loc[pivot_highs_idx].tail(lookback_pivots).values)
    last_lows = list(lows.loc[pivot_lows_idx].tail(lookback_pivots).values)

    if len(last_highs) >= 2 and len(last_lows) >= 2:
        highs_inc = all(x < y for x, y in zip(last_highs, last_highs[1:]))
        lows_inc = all(x < y for x, y in zip(last_lows, last_lows[1:]))
        highs_dec = all(x > y for x, y in zip(last_highs, last_highs[1:]))
        lows_dec = all(x > y for x, y in zip(last_lows, last_lows[1:]))

        if highs_inc and lows_inc:
            return 'HH_HL'
        if highs_dec and lows_dec:
            return 'LL_LH'
        return 'MIXED'
    else:
        return 'INSUFFICIENT_PIVOTS'


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for MTF analysis."""
    df = df.copy()
    df['EMA21'] = ema(df['Close'], 21)
    df['EMA50'] = ema(df['Close'], 50)
    df['MACD_line'], df['MACD_signal'], df['MACD_hist'] = macd(df['Close'])
    df['RSI14'] = rsi(df['Close'], 14)
    df['EMA21_above_EMA50'] = (df['EMA21'] > df['EMA50']).astype(int)
    df['EMA_cross'] = df['EMA21_above_EMA50'].diff().fillna(0)
    df['EMA_cross_signal'] = df['EMA_cross'].apply(lambda x: 1 if x == 1 else (-1 if x == -1 else 0))
    return df


def timeframe_analysis(df_tf: pd.DataFrame) -> Dict:
    """Returns dictionary with indicator states for the timeframe."""
    out = {}
    last = df_tf.iloc[-1]
    out['EMA21'] = last['EMA21']
    out['EMA50'] = last['EMA50']
    out['ema_trend'] = 'bull' if last['EMA21'] > last['EMA50'] else ('bear' if last['EMA21'] < last['EMA50'] else 'flat')
    out['RSI14'] = last['RSI14']
    out['rsi_regime'] = 'overbought' if last['RSI14'] > 70 else ('bull' if last['RSI14'] > 50 else 'bear')
    out['MACD_hist'] = last['MACD_hist']
    out['macd_bias'] = 'bull' if last['MACD_hist'] > 0 else 'bear'
    out['structure'] = hh_hl_structure_from_pivots(df_tf)
    
    recent_cross = df_tf['EMA_cross_signal'].iloc[-3:].sum()
    if df_tf['EMA_cross_signal'].iloc[-3:].max() == 1:
        out['recent_ema_cross'] = 1
    elif df_tf['EMA_cross_signal'].iloc[-3:].min() == -1:
        out['recent_ema_cross'] = -1
    else:
        out['recent_ema_cross'] = 0
    
    out['close_vs_ema21_pct'] = (last['Close'] - last['EMA21']) / last['EMA21'] * 100.0
    return out


def aggregate_signal(daily: Dict, hourly: Dict, lower: Dict) -> Dict:
    """Aggregates rules into BUY/SELL/HOLD and confidence."""
    score = 0
    reasons = []

    if daily['ema_trend'] == 'bull' and daily['macd_bias'] == 'bull':
        score += 40
        reasons.append('Daily bullish (EMA+MACD)')
    elif daily['ema_trend'] == 'bear' and daily['macd_bias'] == 'bear':
        score -= 40
        reasons.append('Daily bearish (EMA+MACD)')
    else:
        reasons.append('Daily neutral/mixed')

    if hourly['ema_trend'] == 'bull' and hourly['macd_bias'] == 'bull':
        score += 25
        reasons.append('Hourly bullish')
    elif hourly['ema_trend'] == 'bear' and hourly['macd_bias'] == 'bear':
        score -= 25
        reasons.append('Hourly bearish')
    else:
        reasons.append('Hourly neutral/mixed')

    if lower['recent_ema_cross'] == 1 and lower['close_vs_ema21_pct'] <= 1.5:
        score += 25
        reasons.append('Lower TF bullish cross with pullback')
    elif lower['recent_ema_cross'] == -1 and lower['close_vs_ema21_pct'] >= -1.5:
        score -= 25
        reasons.append('Lower TF bearish cross with pullback')
    else:
        reasons.append('Lower TF no clear entry cross')

    if daily['structure'] == 'HH_HL':
        score += 5
    if daily['structure'] == 'LL_LH':
        score -= 5

    score = max(-100, min(100, score))
    conf = int(abs(score))

    if score > 0:
        final = 'BUY'
    elif score < 0:
        final = 'SELL'
    else:
        final = 'HOLD'

    return {
        'signal': final,
        'confidence': conf,
        'score': score,
        'reasons': reasons
    }


def mtf_trend_aggregation_engine(df: pd.DataFrame, timeframe_base: str = '1T') -> Dict:
    """Main MTF engine function."""
    df = ensure_datetime_index(df)

    rules = {
        'Daily': 'D',
        '1H': '60T',
        '15min': '15T',
        '5min': '5T'
    }

    dfs = {}
    analyses = {}

    for key, rule in rules.items():
        df_tf = resample_ohlcv(df, rule)
        df_tf = compute_indicators(df_tf)
        dfs[key] = df_tf
        analyses[key] = timeframe_analysis(df_tf)

    lower_key = '5min' if len(dfs['5min']) > 0 else '15min'
    final = aggregate_signal(analyses['Daily'], analyses['1H'], analyses[lower_key])

    return {
        'dataframes': dfs,
        'analyses': analyses,
        'final_signal': final
    }