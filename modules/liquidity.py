"""Liquidity Scanner Module"""
import pandas as pd


def liquidity_signal(df_daily, df_intraday=None):
    """
    Returns liquidity signal and reasons.
    
    Returns:
        {
          "signal": "...",
          "reasons": [...]
        }
    """

    vol_cols = [c for c in df_daily.columns if c.lower() == "volume"]
    if not vol_cols:
        raise ValueError(f"Daily DF missing volume column. Columns: {df_daily.columns.tolist()}")
    vcol = vol_cols[0]

    df = df_daily.sort_index()

    if len(df) < 2:
        return {
            "signal": "❓ Unknown liquidity",
            "reasons": ["Not enough data for liquidity assessment."]
        }

    avg20 = df[vcol].iloc[:-1].tail(20).mean()
    today_vol = df[vcol].iloc[-1]

    reasons = []

    if avg20 > 0:
        volume_strength = today_vol / avg20
    else:
        volume_strength = None

    problems = 0

    if volume_strength is None or volume_strength < 0.7:
        problems += 1
        reasons.append("Today's volume is weak compared to 20-day average.")

    if df_intraday is not None:
        intraday_vol_cols = [c for c in df_intraday.columns if c.lower() == "volume"]
        if intraday_vol_cols:
            ivcol = intraday_vol_cols[0]
            total_vol = df_intraday[ivcol].sum()
            if total_vol > 0:
                spikes = df_intraday[df_intraday[ivcol] > df_intraday[ivcol].rolling(5, min_periods=1).mean() * 3]
                for v in spikes[ivcol]:
                    if v / total_vol > 0.04:
                        problems += 1
                        reasons.append("Large intraday volume spike detected (unstable liquidity).")
                        break

    if problems == 0:
        signal = "✔ High liquidity → Safe to trade"
    elif problems == 1:
        signal = "⚠ Mixed liquidity → Trade with caution"
    else:
        signal = "✖ Low liquidity → Avoid (Slippage risk)"

    return {
        "signal": signal,
        "reasons": reasons
    }