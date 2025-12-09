"""Master Orchestrator for Complete Trading Report"""
import numpy as np
import pandas as pd
import contextlib
import io
from datetime import datetime, timedelta
from typing import Dict

from .data_fetcher import fetch_yahoo_finance_history, get_latest_intraday_data
from .technical_analysis import apply_technical_analysis
from .support_resistance import find_support_resistance
from .news_analysis import get_news_data_async
from .fundamentals import check_fundamentals
from .mtf_engine import mtf_trend_aggregation_engine
from .swing_trading import swing_trade
from .market_regime import market_regime_engine
from .liquidity import liquidity_signal


async def generate_master_report(symbol: str) -> Dict:
    """Generate comprehensive trading report for a symbol."""
    
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            # 1. DATA FETCHING
<<<<<<< HEAD
            df_daily = fetch_yahoo_finance_history(symbol, period='3y')
=======
            df_daily = fetch_yahoo_finance_history(symbol, period='1y')
>>>>>>> 65596a490f88b7354b84d34d5a52877ece134d6f
            if df_daily.empty:
                return {"error": f"No daily data for {symbol}"}
            
            df_intraday = get_latest_intraday_data(symbol, days_back=5)

            # 2. RUN MODULES
            _, signals, ta_summary = apply_technical_analysis(df_daily)
            last_sig = signals.iloc[-1]

            liq = liquidity_signal(df_daily, df_intraday)
            regime = market_regime_engine(df_daily)

            if not df_intraday.empty:
                mtf = mtf_trend_aggregation_engine(df_intraday)
            else:
                mtf = {
                    'final_signal': {'signal': 'HOLD', 'confidence': 0},
                    'analyses': {'Daily': {}, '1H': {}, '5min': {}}
                }

            # Fetch news with improved error handling
            try:
                news = await get_news_data_async(symbol)
                if not news or not isinstance(news, dict):
                    news = {
                        "overall_sentiment": "Neutral",
                        "overall_sentiment_score": 0,
                        "final_conclusion": "Unable to fetch news data at this time.",
                        "articles": []
                    }
            except Exception as e:
                print(f"News fetch error in master_report: {e}")
                news = {
                    "overall_sentiment": "Neutral",
                    "overall_sentiment_score": 0,
                    "final_conclusion": "Unable to fetch news data at this time.",
                    "articles": []
                }

            fund = check_fundamentals(symbol)

        except Exception as e:
            return {"error": str(e)}

    # Helper functions
    def dir_score(val):
        if not val:
            return 0
        s = str(val).lower()
        if any(k in s for k in ["bull", "up", "long", "positive"]): return 1
        if any(k in s for k in ["bear", "down", "short", "negative"]): return -1
        return 0

    def clamp(a, lo, hi): return max(lo, min(hi, a))

    # Extract MTF parts
    analyses = mtf.get("analyses", {})
    d_trend = analyses.get("Daily", {}).get("ema_trend", "neutral")
    h1_trend = analyses.get("1H", {}).get("ema_trend", "neutral")
    ltf = analyses.get("5min", {}).get("structure", "none")

    raw_mtf = mtf.get("final_signal", "HOLD")
    mtf_sig = raw_mtf["signal"] if isinstance(raw_mtf, dict) else raw_mtf

    # LAYER 0 — SAFETY GUARDS
    safety_block = False
    safety_reasons = []

    liq_msg = str(liq["signal"]).lower()
    if any(w in liq_msg for w in ["avoid", "thin", "illiquid"]):
        safety_reasons.append("Thin Liquidity")
        safety_block = True

    regime_msg = str(regime).lower()
    if any(w in regime_msg for w in ["extreme", "panic", "flash", "unstable"]):
        safety_reasons.append("Extreme Volatility")
        safety_block = True

    try:
        prev_close = df_daily["Close"].iloc[-2]
        today_open = df_daily["Open"].iloc[-1]
        gap = abs(today_open - prev_close) / prev_close
        if gap > 0.035:
            safety_reasons.append("Gap Risk")
            gap_risk = True
        else:
            gap_risk = False
    except:
        gap_risk = False

    # LAYER 1 — STRUCTURAL BIAS
    structural_bias = 0
    fscore = float(fund.get("Score (%)", 0) or 0)
    structural_bias += (fscore - 50) / 20
    structural_bias += 1 * dir_score(d_trend)

    # LAYER 2 — TACTICAL ENGINE
    tactical = 0
    tactical += 3 * dir_score(d_trend)
    tactical += 2 * dir_score(h1_trend)
    tactical += 1 * dir_score(ltf)
    tactical += 3 * dir_score(mtf_sig)

    ta_sig = str(ta_summary.get("final_signal", "")).upper()
    if ta_sig == "BUY": tactical += 2
    elif ta_sig == "SELL": tactical -= 2

    tactical += clamp(ta_summary.get("score", 0) * 3, -3, 3)

    # LAYER 3 — CATALYST OVERRIDE
    catalyst_adj = 0
    news_sent = str(news.get("overall_sentiment", "")).lower()

    if any(w in news_sent for w in ["positive", "bull", "optimistic"]):
        catalyst_adj += 3
    if any(w in news_sent for w in ["negative", "bear", "pessimistic"]):
        catalyst_adj -= 3

    try:
        catalyst_adj += clamp(float(news.get("overall_sentiment_score", 0)) * 1.5, -2, 2)
    except:
        pass

    if not df_intraday.empty:
        intr_vol = df_intraday["Volume"].iloc[-1]
        avg_vol = df_intraday["Volume"].tail(60).mean()
        if intr_vol > 3 * avg_vol:
            catalyst_adj += 2 * np.sign(intr_vol - avg_vol)

    # COMBINATION: FINAL SCORE
    final_score = structural_bias + tactical + catalyst_adj

    if safety_block and abs(catalyst_adj) < 3:
        final_act = "HOLD"
    else:
        if final_score >= 2.0:
            final_act = "BUY"
        elif final_score <= -2.0:
            final_act = "SELL"
        else:
            final_act = "HOLD"

    if final_act == "SELL" and catalyst_adj >= 3:
        final_act = "HOLD"

    if final_act == "BUY" and catalyst_adj <= -3:
        final_act = "HOLD"

    # CONFIDENCE MODEL
    magnitude = abs(final_score)

    if final_act == "HOLD":
        conf = 35 + min(10, magnitude * 3)
        if safety_block: conf += 5
    else:
        conf = 55 + min(25, magnitude * 5)
        if gap_risk: conf -= 7
        if safety_block: conf -= 7

    final_conf = int(clamp(conf, 20, 95))

    # SWING TRADE
    swing_data = None
    if final_act in ["BUY", "HOLD"]:
        try:
            use_df = df_intraday if not df_intraday.empty else df_daily
            swing_data = swing_trade(use_df, symbol, product="margin", risk_pct=0.01, rr=2.0)
            
            if swing_data["Score"] == 0:
                swing_data["Score"] = max(20, final_conf - 20)
        except:
            last_close = df_daily["Close"].iloc[-1]
            swing_data = {
                "Symbol": symbol,
                "Entry": round(last_close, 2),
                "Stoploss": round(last_close * 0.95, 2),
                "Target": round(last_close * 1.10, 2),
                "Score": 30,
                "Reasons": ["Basic swing setup based on current price"]
            }

    # SUPPORT / RESISTANCE
    try:
        sr = find_support_resistance(df_daily)
    except:
        sr = {
            'nearest_resistances': [],
            'nearest_supports': [],
            'resistance_zones': [],
            'support_zones': []
        }

    # Return complete report data
    return {
        "symbol": symbol,
        "final_action": final_act,
        "confidence": final_conf,
        "final_score": final_score,
        "safety_warnings": safety_reasons,
        "technical_analysis": {
            "summary": ta_summary,
            "signals": last_sig.to_dict()
        },
        "support_resistance": sr,
        "liquidity": liq,
        "regime": regime,
        "mtf": {
            "daily_trend": d_trend,
            "hourly_trend": h1_trend,
            "lower_tf_structure": ltf,
            "final_signal": mtf_sig,
            "confidence": final_conf
        },
        "news": news,
        "fundamentals": fund,
        "swing_trade": swing_data,
        "df_daily": df_daily,
        "df_intraday": df_intraday
    }