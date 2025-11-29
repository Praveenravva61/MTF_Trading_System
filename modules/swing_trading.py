"""Swing Trading Setup Module"""
import pandas as pd
import numpy as np


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14):
    """Average True Range."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def swing_entry(df_daily, df_hourly, rr=2.0):
    """Calculate swing entry setup."""
    df_daily = df_daily.copy()
    df_hourly = df_hourly.copy()

    df_daily['EMA21'] = ema(df_daily['Close'], 21)
    df_daily['ATR14'] = atr(df_daily, 14)

    df_hourly['EMA21'] = ema(df_hourly['Close'], 21)
    df_hourly['ATR14'] = atr(df_hourly, 14)

    last_d = df_daily.iloc[-1]
    last_h = df_hourly.iloc[-1]

    entry = float(last_h["Close"])
    atr_d = float(last_d["ATR14"])

    reasons = []
    score = 0

    def bullish_engulf(df):
        if len(df) < 2:
            return False
        p = df.iloc[-2]
        c = df.iloc[-1]
        return (
            p["Close"] < p["Open"] and
            c["Close"] > c["Open"] and
            c["Open"] < p["Close"] and
            c["Close"] > p["Open"]
        )
    
    def hammer(df):
        c = df.iloc[-1]
        body = abs(c["Close"] - c["Open"])
        lw = min(c["Open"], c["Close"]) - c["Low"]
        return lw > 2 * body

    if bullish_engulf(df_daily):
        score += 20
        reasons.append("Bullish Engulfing (Daily)")

    if hammer(df_daily):
        score += 15
        reasons.append("Hammer Pattern (Daily)")

    if last_d["Close"] > last_d["EMA21"]:
        score += 10
        reasons.append("Daily close above EMA21")

    swing_high = float(df_hourly["High"].iloc[:-2].max())
    breakout = entry > swing_high

    if breakout:
        score += 25
        reasons.append(f"Breakout above {round(swing_high,2)} (Hourly)")

        retest = abs(df_hourly['Low'].iloc[-1] - swing_high) <= (0.005 * swing_high)
        if retest:
            score += 20
            reasons.append("Retest successful (Hourly)")
    else:
        retest = False

    pullback = abs(entry - last_d["EMA21"]) <= 0.015 * last_d["EMA21"]

    if pullback:
        score += 20
        reasons.append("Pullback near Daily EMA21")

    if breakout and retest:
        sl = swing_high - 1.5 * atr_d
    elif pullback:
        sl = last_d["EMA21"] - 1.5 * atr_d
    else:
        recent_low = float(df_daily["Low"].iloc[-3:].min())
        sl = recent_low - 0.5 * atr_d

    if sl >= entry:
        sl = entry - atr_d
        reasons.append("SL auto-adjusted using ATR")

    target = entry + rr * (entry - sl)
    per_share_risk = entry - sl

    score = max(0, min(100, score))

    return {
        "entry": entry,
        "sl": sl,
        "target": target,
        "per_share_risk": per_share_risk,
        "score": score,
        "reasons": reasons
    }


def swing_trade(df1, symbol, product="margin", risk_pct=0.01, rr=2.0):
    """Main swing trade function."""
    df1 = df1.copy()
    df1.index = pd.to_datetime(df1.index)

    df_daily = df1.resample("D").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
    df_hourly = df1.resample("60T").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()

    if df_hourly.empty:
        df_hourly = df_daily.copy()

    analysis = swing_entry(df_daily, df_hourly, rr=rr)

    return {
        "Symbol": symbol,
        "Entry": round(analysis["entry"], 2),
        "Stoploss": round(analysis["sl"], 2),
        "Target": round(analysis["target"], 2),
        "Score": analysis["score"],
        "Reasons": analysis["reasons"],
    }