"""Market Regime and Volatility Detection Module"""
import pandas as pd
import numpy as np


def market_regime_engine(df, vol_lookback=20, choppy_lookback=14, atr_lookback=14):
    """
    Detects High Volatility, Choppy Market (No Trade), or OK Market.
    
    Returns:
        str: one of the following:
            - "high volatility: High chances of Hitting stoploss"
            - "Choppy market → NO TRADE"
            - "Market condition is ok"
    """

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(atr_lookback).mean()
    atr_pct = atr / close * 100

    recent_atr_pct = atr_pct.iloc[-1]

    up = high.diff()
    down = low.diff() * -1

    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)

    tr_smooth = tr.rolling(choppy_lookback).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(choppy_lookback).sum() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm).rolling(choppy_lookback).sum() / tr_smooth)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(choppy_lookback).mean()

    recent_adx = adx.iloc[-1]

    if recent_atr_pct > 2:
        return "high volatility: High chances of Hitting stoploss"

    if recent_adx < 20:
        return "Choppy market → NO TRADE"

    return "Market condition is ok"