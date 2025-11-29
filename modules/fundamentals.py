"""Fundamental Analysis Module"""
import yfinance as yf


def check_fundamentals(ticker: str):
    """Check if a stock has strong fundamentals using Yahoo Finance data."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        roe = info.get("returnOnEquity")
        roa = info.get("returnOnAssets")
        profit_margin = info.get("profitMargins")
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        market_cap = info.get("marketCap")

        results = {
            "PE Ratio": pe,
            "PB Ratio": pb,
            "ROE": roe,
            "ROA": roa,
            "Profit Margin": profit_margin,
            "Debt to Equity": debt_to_equity,
            "Current Ratio": current_ratio,
            "Quick Ratio": quick_ratio,
            "Market Cap": market_cap
        }

        score = 0
        rules = 0

        def check(condition):
            nonlocal score, rules
            rules += 1
            if condition:
                score += 1

        check(pe is not None and 5 < pe < 25)
        check(pb is not None and pb < 4)
        check(roe is not None and roe > 0.12)
        check(roa is not None and roa > 0.05)
        check(profit_margin is not None and profit_margin > 0.08)
        check(debt_to_equity is not None and debt_to_equity < 100)
        check(current_ratio is not None and current_ratio > 1.2)
        check(quick_ratio is not None and quick_ratio > 1)
        check(market_cap is not None and market_cap > 2e9)

        strength = (score / rules) * 100 if rules > 0 else 0

        if strength >= 75:
            verdict = "Strong"
            emoji = "⭐"
        elif 45 <= strength < 75:
            verdict = "Average"
            emoji = "⚠️"
        else:
            verdict = "Weak"
            emoji = "❌"

        return {
            "Ticker": ticker,
            "Score (%)": round(strength, 2),
            "Verdict": verdict,
            "Emoji": emoji,
            "Raw Data": results
        }
    except Exception as e:
        return {
            "Ticker": ticker,
            "Score (%)": 0,
            "Verdict": "Unavailable",
            "Emoji": "⚠️",
            "Raw Data": {}
        }