"""MTF Trading System - Advanced Dynamic Streamlit Application with Dynamic Backgrounds"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import random
import os
import base64
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Import modules
from modules.master_report import generate_master_report
from modules.load_models import load_forecast_assets,save_forecast_assets
from modules.data_fetcher import get_stock_info 
from modules.forecasting import train_forecasting_model,forecast_future_with_ci, forecast_future, plot_future, evaluate_model, fetch_data, plot_forecast
from utils.visualizations import (
    create_candlestick_chart,
    create_indicator_chart,
    create_macd_chart,
    create_intraday_chart,
    create_gauge_chart,
    create_mtf_trend_chart,
    create_volume_analysis_chart,
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MTF Trading System v0.01",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# DYNAMIC BACKGROUND LOADING
# =========================================================
def get_random_background_image():
    """Load a random image from the specified path as base64."""
    image_path = r"A:\\PROJECTS\\MTF_TRADING_SYSTEM\\Images"
    
    try:
        if os.path.exists(image_path):
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(Path(image_path).glob(f'*{ext}'))
                image_files.extend(Path(image_path).glob(f'*{ext.upper()}'))
            
            if image_files:
                # Select random image
                selected_image = random.choice(image_files)
                
                # Read and encode to base64
                with open(selected_image, 'rb') as f:
                    image_data = f.read()
                    encoded = base64.b64encode(image_data).decode()
                    
                # Determine mime type
                ext = selected_image.suffix.lower()
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(ext, 'image/jpeg')
                
                return f"data:{mime_type};base64,{encoded}"
        
        # Fallback to gradient if no images found
        return None
        
    except Exception as e:
        print(f"Error loading background image: {e}")
        return None

# Try to load background image, fallback to gradient
background_image = get_random_background_image()

BACKGROUND_GRADIENTS = [
    "radial-gradient(circle at top left, #0f172a 0, #020617 45%, #000000 100%)",
    "radial-gradient(circle at top, #111827 0, #020617 50%, #0b1120 100%)",
    "radial-gradient(circle at top right, #020617 0, #1f2933 40%, #020617 100%)",
    "linear-gradient(135deg, #020617 0%, #0f172a 40%, #111827 100%)",
]

if background_image:
    selected_bg = f"linear-gradient(rgba(2, 6, 23, 0.85), rgba(2, 6, 23, 0.85)), url('{background_image}')"
    bg_style = f"background: {selected_bg}; background-size: cover; background-position: center; background-attachment: fixed;"
else:
    selected_bg = random.choice(BACKGROUND_GRADIENTS)
    bg_style = f"background: {selected_bg};"

TRADING_QUOTES = [
    ("The trend is your friend until it bends.", "– Trading Proverb"),
    ("Risk comes from not knowing what you are doing.", "– Warren Buffett"),
    ("Cut your losses short and let your winners run.", "– David Ricardo"),
    ("An edge with discipline beats prediction with emotion.", "– Unknown"),
    ("Plan the trade and trade the plan.", "– Trading Rule"),
    ("Amateurs think about how much they can make; professionals think about how much they can lose.", "– Jack Schwager"),
]

selected_quote, selected_quote_author = random.choice(TRADING_QUOTES)

# =========================================================
# GLOBAL CSS WITH DYNAMIC BACKGROUND
# =========================================================
st.markdown(
    f"""
<style>
body, .stApp {{
    {bg_style}
}}

/* ================= SIDEBAR STYLING ================= */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0B1120 0%, #111827 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
}}

/* Sidebar Content Container */
.sidebar-content {{
    padding: 1rem 0;
}}

.sidebar-header {{
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}

.sidebar-subtext {{
    color: #94a3b8;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    line-height: 1.4;
}}

.sidebar-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.3), transparent);
    margin: 1.5rem 0;
}}

/* Custom Inputs in Sidebar */
.stSelectbox > div > div {{
    background-color: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: white;
}}
.stTextInput > div > div > input {{
    background-color: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: white;
}}

/* ================= MAIN CONTENT STYLING ================= */

/* Main header */
.main-header {{
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #22d3ee 0%, #818cf8 50%, #f97316 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
    text-shadow: 0 0 30px rgba(34, 211, 238, 0.3);
}}

h2, h3 {{
    font-weight: 700;
    color: #e5e7eb;
}}

/* Detailed Metrics Tab Cards */
.detail-card {{
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}}
.detail-card:hover {{
    background: rgba(30, 41, 59, 0.9);
    border-color: rgba(56, 189, 248, 0.3);
    transform: translateY(-2px);
}}
.detail-label {{
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}}
.detail-value {{
    color: #f1f5f9;
    font-size: 1.3rem;
    font-weight: 600;
}}
.detail-sub {{
    font-size: 0.85rem;
    margin-top: 0.25rem;
}}

.metric-card {{
    background: linear-gradient(135deg, rgba(148,163,184,0.15), rgba(30,64,175,0.7));
    padding: 1.25rem;
    border-radius: 14px;
    color: #e5e7eb;
    text-align: center;
    box-shadow: 0 18px 35px rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.35);
    backdrop-filter: blur(15px);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease;
}}
.metric-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 24px 45px rgba(15,23,42,0.8);
    border-color: #60a5fa;
}}

.signal-buy, .signal-sell, .signal-hold {{
    padding: 1.1rem 1rem;
    border-radius: 16px;
    color: white;
    font-size: 1.6rem;
    font-weight: 800;
    text-align: center;
    margin: 0.4rem 0 1.0rem 0;
    box-shadow: 0 22px 40px rgba(15,23,42,0.9);
    backdrop-filter: blur(10px);
}}
.signal-buy {{
    background: radial-gradient(circle at top left, #22c55e 0%, #16a34a 35%, #166534 100%);
}}
.signal-sell {{
    background: radial-gradient(circle at top left, #f97373 0%, #ef4444 35%, #7f1d1d 100%);
}}
.signal-hold {{
    background: radial-gradient(circle at top left, #94a3b8 0%, #64748b 35%, #020617 100%);
}}

.news-card {{
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,64,175,0.6));
    padding: 0.75rem 1rem;
    border-radius: 10px;
    margin: 0.4rem 0;
    border-left: 4px solid #6366f1;
    color: #e5e7eb;
    backdrop-filter: blur(10px);
}}

.warning-banner {{
    background: rgba(255, 243, 205, 0.95);
    color: #856404;
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    border: 1px solid #ffeeba;
    margin: 0.6rem 0 1.0rem 0;
    backdrop-filter: blur(10px);
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.25rem;
}}
.stTabs [data-baseweb="tab"] {{
    background-color: rgba(15,23,42,0.85);
    border-radius: 999px;
    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
    backdrop-filter: blur(10px);
}}

.block-container {{
    padding-top: 1.0rem;
}}

/* Quote banner */
.quote-banner {{
    background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(37,99,235,0.6));
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin: 0.7rem 0 1.0rem 0;
    color: #e5e7eb;
    box-shadow: 0 18px 40px rgba(15,23,42,0.9);
    border: 1px solid rgba(96,165,250,0.6);
    backdrop-filter: blur(15px);
}}

/* Enhanced visibility for content */
.stMarkdown, .stDataFrame, .stMetric {{
    backdrop-filter: blur(5px);
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# POPULAR STOCKS
# =========================================================
POPULAR_STOCKS = {
    "Select a stock": "",
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Larsen & Toubro": "LT.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Wipro": "WIPRO.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Titan Company": "TITAN.NS",
    "Power Grid": "POWERGRID.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "NTPC": "NTPC.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tech Mahindra": "TECHM.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Dr Reddy's": "DRREDDY.NS",
    "Cipla": "CIPLA.NS",
    "Grasim": "GRASIM.NS",
    "HCL Tech": "HCLTECH.NS",
    "Punjab National Bank": "PNB.NS",
    "NCC Limited": "NCC.NS",
    "Suzlon Energy": "SUZLON.NS",
}

# =========================================================
# HEADER
# =========================================================
st.markdown('<h1 class="main-header">📈 MTF Trading System v0.01</h1>', unsafe_allow_html=True)
st.markdown(
    "### Multi-Timeframe Trading Desk — **Final Signal, Entry, Stoploss & Target at a Glance**"
)

# =========================================================
# SIDEBAR (Classy Look)
# =========================================================
with st.sidebar:
    # Attempt to load a logo if available in the path, else text
    logo_path = Path(r"A:\\PROJECTS\\MTF_TRADING_SYSTEM\\Images\\/images/Logo.jpg") 
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    # Classy Header
    st.markdown(
        """
        <div style="text-align: center; padding-bottom: 20px;">
             <h2 class="sidebar-header">MTF SYSTEM</h2>
             <p class="sidebar-subtext">Professional Grade<br>Algorithmic Analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 🛠️ Control Panel")
    
    selected_name = st.selectbox(
        "Select Asset",
        options=list(POPULAR_STOCKS.keys()),
        index=0,
    )

    custom_ticker = st.text_input("Or Custom Ticker", placeholder="e.g. RELIANCE.NS")

    if custom_ticker:
        ticker = custom_ticker.upper()
    else:
        ticker = POPULAR_STOCKS[selected_name]
    
    st.write("") # Spacer
    analyze_button = st.button("🚀 START ANALYSIS", type="primary", width='stretch')

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown("### 📘 User Guide")
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #cbd5e1;">
            <div style="margin-bottom: 8px;"><b>1. Select Asset</b><br>Choose from list or type manually.</div>
            <div style="margin-bottom: 8px;"><b>2. Run Analysis</b><br>Click the Start button above.</div>
            <div style="margin-bottom: 8px;"><b>3. Interpret</b><br>Check Signal, Confidence & Regime.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    
    st.warning("⚠️ Educational Use Only")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# MAIN CONTENT
# =========================================================
if not ticker:
    # Landing page with dynamic quote
    st.info("👈 Select a stock from the left sidebar to start analysis.")

    st.markdown(
        f"""
        <div class="quote-banner">
            <h3>💡 Trading Insight</h3>
            <p style="font-size:1.1rem;font-style:italic;">"{selected_quote}"</p>
            <p style="text-align:right;font-size:0.95rem;">{selected_quote_author}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>📊 Multi-Timeframe Edge</h3>
                <p>Combine Daily, Hourly and Intraday signals for cleaner decisions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🎯 Risk First</h3>
                <p>Every setup comes with Entry, Stoploss, Target & R:R.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>🧠 Regime + Liquidity</h3>
                <p>Detect choppy / high-volatility phases and avoid bad trades.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    if analyze_button or st.session_state.get("last_ticker") == ticker:
        st.session_state["last_ticker"] = ticker

        with st.spinner(f"🔍 Analyzing **{ticker}** across multiple timeframes..."):
            report = asyncio.run(generate_master_report(ticker))

        if "error" in report:
            st.error(f"❌ Error: {report['error']}")
        else:
            # ==========================
            # STOCK INFO HEADER
            # ==========================
            try:
                stock_info = get_stock_info(ticker)
                display_name = stock_info.get("Name", ticker)
            except Exception:
                stock_info = {}
                display_name = ticker

            st.markdown(f"## 🧾 {display_name}  ({ticker})")

            info_cols = st.columns(4)
            with info_cols[0]:
                st.metric("Current Price", f"₹{stock_info.get('Current Price', 'N/A')}")
            with info_cols[1]:
                st.metric("Sector", stock_info.get("Sector", "N/A"))
            with info_cols[2]:
                st.metric("Industry", stock_info.get("Industry", "N/A"))
            with info_cols[3]:
                mc = stock_info.get("Market Cap", 0)
                if isinstance(mc, (int, float)) and mc > 0:
                    st.metric("Market Cap", f"₹{mc/1e9:.2f}B")
                else:
                    st.metric("Market Cap", "N/A")

            # Safety warnings
            if report.get("safety_warnings"):
                warning_text = " | ".join(report["safety_warnings"])
                st.markdown(
                    f"""
                    <div class="warning-banner">
                        ⚠️ <b>RISK WARNING:</b> {warning_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ==========================
            # Pull data from report
            # ==========================
            df_daily = report["df_daily"]
            df_intraday = report["df_intraday"]
            sr = report["support_resistance"]
            ta_data = report["technical_analysis"]
            mtf = report["mtf"]
            swing = report.get("swing_trade")
            liq = report.get("liquidity", {})
            regime = report.get("regime", "Market condition status unavailable")
            news_raw = report.get("news", {})

            final_action = report["final_action"]
            confidence = report["confidence"]

            # ====================================================
            # 1) DECISION PANEL — FINAL SIGNAL + ENTRY/SL/TARGET
            # ====================================================
            st.markdown("## 🧭 Decision Panel — First Read This")

            top_left, top_right = st.columns([1.5, 1.2])

            with top_left:
                # Final Signal card
                if final_action == "BUY":
                    st.markdown(
                        f"""
                        <div class="signal-buy">
                            🟢 FINAL ACTION: BUY<br>
                            <span style="font-size:0.9em;font-weight:500;">Confidence: {confidence}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif final_action == "SELL":
                    st.markdown(
                        f"""
                        <div class="signal-sell">
                            🔴 FINAL ACTION: SELL<br>
                            <span style="font-size:0.9em;font-weight:500;">Confidence: {confidence}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="signal-hold">
                            ⚪ FINAL ACTION: HOLD / NO TRADE<br>
                            <span style="font-size:0.9em;font-weight:500;">Confidence: {confidence}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.plotly_chart(
                    create_gauge_chart(confidence, "Signal Confidence"),
                    width='stretch',
                )

            with top_right:
                # Show Entry / SL / Target prominently if swing is available
                st.markdown("#### 🎯 Key Trade Levels")
                if swing:
                    el, sl, tl, rr = st.columns(4)
                    with el:
                        st.metric("Entry", f"₹{swing['Entry']}")
                    with sl:
                        st.metric("Stoploss", f"₹{swing['Stoploss']}")
                    with tl:
                        st.metric("Target", f"₹{swing['Target']}")
                    with rr:
                        # quick R:R calculation
                        risk = swing["Entry"] - swing["Stoploss"]
                        reward = swing["Target"] - swing["Entry"]
                        rr_ratio = reward / risk if risk > 0 else 0
                        st.metric("R:R", f"1:{rr_ratio:.2f}")

                    st.caption(f"Setup Score: **{swing['Score']}/100**")
                else:
                    st.info(
                        "No long swing setup generated for this symbol. Check Technicals & Regime before trading."
                    )

                st.markdown("#### 🧊 Regime & Liquidity Snapshot")
                if "high volatility" in regime.lower():
                    st.error(f"⚠️ {regime}")
                elif "choppy" in regime.lower():
                    st.warning(f"⚠️ {regime}")
                else:
                    st.success(f"✅ {regime}")

                if liq:
                    st.info(liq.get("signal", "Liquidity info unavailable"))
                else:
                    st.info("Liquidity info unavailable")

            st.markdown("---")
            # ====================================================
            # 1.5) AI FORECASTING AGENT (UPDATED)
            # ====================================================
            st.markdown("## 🔮 AI Forecasting Agent (60-Day Prediction)")

            with st.expander("✨ Run Deep Learning Forecast (CNN + LSTM)", expanded=False):

                st.info("This module trains an optimized forecasting model and predicts the next 60 days. Faster & more accurate.")

                if st.button("🚀 Run Deep Forecast", key="btn_forecast"):
                    with st.spinner("🤖 Training model & generating forecast..."):

                        try:
                            # 0️⃣ Check if model already exists
                            model, df_processed, actual_next, pred_next,residual_std = load_forecast_assets(ticker)

                            if model is None:
                                st.info("📦 No saved model found — training new model...")

                                # 🟦 1) TRAIN MODEL (NEW)
                                df_raw = fetch_data(ticker)
                                model, df_processed, actual_next, pred_next, residual_std = train_forecasting_model(df_raw)

# use actual_next & pred_next in your "Model Validation: Actual vs Predicted" chart


                                # 🟦 2) SAVE THE MODEL & DATA
                                save_forecast_assets(ticker, model, df_processed, actual_next, pred_next,residual_std)

                                st.success("🎉 Model trained and saved successfully!")
                            else:   
                                st.success("⚡ Loaded cached model (no retraining needed)")


                            # -----------------------------------------
                            #  DISPLAY METRICS IN 3 COLUMNS
                            # -----------------------------------------
                            st.markdown("### 📊 Model Summary")

                            metrics = evaluate_model(df_processed, actual_next, pred_next, model, runs=50)

                            m1, m2, m3 = st.columns(3)

                            with m1:
                                st.metric("RMSE", f"{metrics['rmse']:.2f}")

                            with m2:
                                st.metric("MAE", f"{metrics['mae']:.2f}")

                            with m3:
                                st.metric("Direction Accuracy", f"{metrics['direction_accuracy']:.1f}%")

                            st.metric("Forecast Confidence", f"{metrics['forecast_confidence']:.1f}%")

                            # -----------------------------------------
                            #  SHOW FORECAST CHART
                            # -----------------------------------------
                            st.markdown("### 🔮 Price Forecast")
                            fig_val = plot_forecast(df_processed, actual_next, pred_next)
                            st.plotly_chart(fig_val, use_container_width=True)
                            
                            st.markdown("### 🔮 Next 60 Days Forecast")
                            result = forecast_future_with_ci(model, df_processed, days=60, residual_std=residual_std)
                            forecast_df = result["forecast_df"]

                            
                            fig_future = plot_future(df_processed, forecast_df)
                            st.plotly_chart(fig_future, use_container_width=True)

                            # -----------------------------------------
                            # FORECAST DATA TABLE
                            # -----------------------------------------
                            with st.expander("📋 Detailed Forecast Values"):

                                try:
                                    display_df = pd.DataFrame({
                                        "Date": forecast_df.index,
                                        "Predicted Price": forecast_df["Close_Pred"].values
                                    })

                                    st.dataframe(
                                        display_df.style.format({"Predicted Price": "₹{:.2f}"}),
                                        use_container_width=True,
                                        hide_index=True
                                    )

                                except Exception as e:
                                    st.error(f"❌ Error running forecast: {str(e)}")
                                    st.exception(e)
                        except Exception as e:
                            st.error(f"❌ Error running forecast: {str(e)}")
                            st.exception(e)



            # ====================================================
            # 2) CHART ROW — DAILY + INTRADAY
            # ====================================================
            st.markdown("## 📊 Price Action Charts")

            charts_left, charts_right = st.columns([1.2, 1])

            with charts_left:
                daily_fig = create_candlestick_chart(
                    df_daily.tail(100), f"{ticker} — Daily Candlestick", True, sr
                )
                daily_fig.update_layout(height=520)
                st.plotly_chart(daily_fig, width='stretch')

            with charts_right:
                if df_intraday is None or df_intraday.empty:
                    st.warning("Intraday data not available for today.")
                else:
                    intraday_fig = create_intraday_chart(
                        df_intraday, f"{ticker} — Intraday Movement"
                    )
                    closes = df_intraday["Close"].dropna()
                    if not closes.empty:
                        min_p = float(closes.min())
                        max_p = float(closes.max())
                        if min_p == max_p:
                            pad = max_p * 0.005 if max_p != 0 else 1
                            y_min, y_max = max_p - pad, max_p + pad
                        else:
                            span = max_p - min_p
                            pad = span * 0.2
                            y_min, y_max = min_p - pad, max_p + pad
                        intraday_fig.update_yaxes(range=[y_min, y_max])
                    intraday_fig.update_layout(height=520)
                    st.plotly_chart(intraday_fig, width='stretch')

            st.markdown("---")

            # ====================================================
            # 3) TABS — TECHNICALS / SWING / NEWS / FUND / DETAILS
            # ====================================================
            tab_ta, tab_swing, tab_news, tab_fund, tab_details = st.tabs(
                [
                    "📈 Technicals & Volume",
                    "🎯 Swing Trade Details",
                    "📰 News & Sentiment",
                    "💼 Fundamentals",
                    "🔍 Detailed Metrics",
                ]
            )

            # ---------- TECHNICALS TAB ----------
            with tab_ta:
                st.markdown("### 📈 Technical Dashboard")

                summary = ta_data["summary"]
                signals = ta_data["signals"]

                tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                with tcol1:
                    rsi_val = summary["indicators"]["RSI"]
                    rsi_color = "🟢" if rsi_val < 30 else "🔴" if rsi_val > 70 else "🟡"
                    st.metric("RSI (14)", f"{rsi_val:.2f} {rsi_color}")
                with tcol2:
                    macd_val = summary["indicators"]["MACD"]
                    macd_color = "🟢" if macd_val > 0 else "🔴"
                    st.metric("MACD", f"{macd_val:.2f} {macd_color}")
                with tcol3:
                    adx_val = summary["indicators"]["ADX"]
                    adx_color = "🟢" if adx_val > 25 else "🟡"
                    st.metric("ADX", f"{adx_val:.2f} {adx_color}")
                with tcol4:
                    st.metric("Momentum", summary["momentum"])

                st.markdown("#### 📊 Indicator Signals")
                signal_data = {
                    "Indicator": ["SMA", "EMA", "MACD", "RSI", "Bollinger", "Stochastic", "OBV", "ADX"],
                    "Signal": [
                        "🔼 Bullish" if signals["sma"] > 0 else "🔻 Bearish" if signals["sma"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["ema"] > 0 else "🔻 Bearish" if signals["ema"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["macd"] > 0 else "🔻 Bearish" if signals["macd"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["rsi"] > 0 else "🔻 Bearish" if signals["rsi"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["boll"] > 0 else "🔻 Bearish" if signals["boll"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["stoch"] > 0 else "🔻 Bearish" if signals["stoch"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["obv"] > 0 else "🔻 Bearish" if signals["obv"] < 0 else "⚪ Neutral",
                        "🔼 Bullish" if signals["adx"] > 0 else "🔻 Bearish" if signals["adx"] < 0 else "⚪ Neutral",
                    ],
                    "Value": [
                        signals["sma"],
                        signals["ema"],
                        signals["macd"],
                        signals["rsi"],
                        signals["boll"],
                        signals["stoch"],
                        signals["obv"],
                        signals["adx"],
                    ],
                }
                st.dataframe(pd.DataFrame(signal_data), width='stretch', hide_index=True)

                st.markdown("#### 🕒 Multi-Timeframe Trend Grid")
                st.plotly_chart(create_mtf_trend_chart(mtf), width='stretch')

                st.markdown("#### 📊 Volume Context")
                st.plotly_chart(
                    create_volume_analysis_chart(df_daily.tail(100)),
                    width='stretch',
                )

            # ---------- SWING TRADE TAB ----------
            with tab_swing:
                st.markdown("### 🎯 Swing Trade Details")

                if swing:
                    scol1, scol2, scol3, scol4 = st.columns(4)
                    with scol1:
                        st.metric("Entry", f"₹{swing['Entry']}")
                    with scol2:
                        st.metric("Stop Loss", f"₹{swing['Stoploss']}")
                    with scol3:
                        st.metric("Target", f"₹{swing['Target']}")
                    with scol4:
                        st.metric("Setup Score", f"{swing['Score']}/100")

                    risk = swing["Entry"] - swing["Stoploss"]
                    reward = swing["Target"] - swing["Entry"]
                    rr_ratio = reward / risk if risk > 0 else 0

                    st.markdown("#### 📊 Risk–Reward Breakdown")
                    rr1, rr2, rr3 = st.columns(3)
                    with rr1:
                        st.metric("Risk / Share", f"₹{risk:.2f}")
                    with rr2:
                        st.metric("Reward / Share", f"₹{reward:.2f}")
                    with rr3:
                        st.metric("R:R Ratio", f"1:{rr_ratio:.2f}")

                    st.markdown("#### 💡 Setup Reasons")
                    for reason in swing["Reasons"]:
                        st.markdown(f"• {reason}")

                    st.markdown("#### 🧱 Key Levels")
                    k1, k2 = st.columns(2)
                    with k1:
                        st.markdown("**🔺 Resistance Levels**")
                        for level in sr["nearest_resistances"]:
                            st.markdown(f"• **₹{level}**")
                    with k2:
                        st.markdown("**🔻 Support Levels**")
                        for level in sr["nearest_supports"]:
                            st.markdown(f"• **₹{level}**")
                else:
                    st.info("No swing trade setup for this symbol. Check Technicals & Regime before trading.")

            # ---------- NEWS TAB ----------
            with tab_news:
                st.markdown("### 📰 News & Sentiment")

                if isinstance(news_raw, dict):
                    news = news_raw
                else:
                    news = {
                        "overall_sentiment": "Neutral",
                        "overall_sentiment_score": 0,
                        "final_conclusion": str(news_raw),
                        "articles": [],
                    }

                sentiment = news.get("overall_sentiment", "Neutral")
                score = float(news.get("overall_sentiment_score", 0))

                if sentiment.lower() == "bullish":
                    sentiment_color = "#22c55e"
                    sentiment_emoji = "🟢"
                elif sentiment.lower() == "bearish":
                    sentiment_color = "#ef4444"
                    sentiment_emoji = "🔴"
                else:
                    sentiment_color = "#6b7280"
                    sentiment_emoji = "⚪"

                n1, n2 = st.columns(2)
                with n1:
                    st.markdown(
                        f"""
                        <div style="background:{sentiment_color};padding:1.4rem;border-radius:14px;text-align:center;">
                            <h2 style="color:white;margin:0;">{sentiment_emoji} {sentiment}</h2>
                            <p style="color:white;margin:0;">Overall News Bias</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with n2:
                    st.plotly_chart(
                        create_gauge_chart(abs(score) * 100, "Sentiment Strength"),
                        width='stretch',
                    )

                st.markdown("#### 📝 Summary")
                st.info(
                    news.get(
                        "final_conclusion",
                        "Unable to fetch detailed news summary at this time.",
                    )
                )

                st.markdown("#### 📰 Recent Headlines")
                articles = news.get("articles") or []
                if articles:
                    for article in articles[:10]:
                        source = article.get("source", "Unknown")
                        headline = article.get("headline", "No headline")
                        st.markdown(
                            f"""
                            <div class="news-card">
                                <b>{source}</b><br>
                                {headline}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning(
                        "No recent news articles found. This could indicate limited media coverage for this stock."
                    )

            # ---------- FUNDAMENTALS TAB ----------
            with tab_fund:
                st.markdown("### 💼 Fundamental Quality")

                fund = report["fundamentals"]
                verdict = fund["Verdict"]
                emoji = fund["Emoji"]
                score_f = fund["Score (%)"]

                f1, f2, f3 = st.columns([1, 2, 1])
                with f2:
                    if verdict == "Strong":
                        color = "#22c55e"
                    elif verdict == "Average":
                        color = "#facc15"
                    else:
                        color = "#ef4444"

                    st.markdown(
                        f"""
                        <div style="background:{color};padding:2rem;border-radius:14px;text-align:center;">
                            <h1 style="color:white;margin:0;">{emoji} {verdict}</h1>
                            <h3 style="color:white;margin:0;">Fundamental Score: {score_f}%</h3>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("---")

                raw_data = fund["Raw Data"]
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    pe = raw_data.get("PE Ratio")
                    st.metric("PE Ratio", f"{pe:.2f}" if pe else "N/A")
                    pb = raw_data.get("PB Ratio")
                    st.metric("PB Ratio", f"{pb:.2f}" if pb else "N/A")
                with c2:
                    roe = raw_data.get("ROE")
                    st.metric("ROE", f"{roe*100:.2f}%" if roe else "N/A")
                    roa = raw_data.get("ROA")
                    st.metric("ROA", f"{roa*100:.2f}%" if roa else "N/A")
                with c3:
                    pm = raw_data.get("Profit Margin")
                    st.metric("Profit Margin", f"{pm*100:.2f}%" if pm else "N/A")
                    de = raw_data.get("Debt to Equity")
                    st.metric("Debt/Equity", f"{de:.2f}" if de else "N/A")
                with c4:
                    cr = raw_data.get("Current Ratio")
                    st.metric("Current Ratio", f"{cr:.2f}" if cr else "N/A")
                    qr = raw_data.get("Quick Ratio")
                    st.metric("Quick Ratio", f"{qr:.2f}" if qr else "N/A")

            # ---------- DETAILED METRICS TAB (Redesigned) ----------
            with tab_details:
                st.markdown("### 🔍 Advanced Metrics Explorer")
                
                # --- Row 1: Status Cards ---
                st.markdown("#### 💠 Market State Analysis")
                d1, d2 = st.columns(2)
                
                # Liquidity Card
                with d1:
                    liq_sig = liq.get("signal", "N/A")
                    liq_color = "#3b82f6" if "Liquid" in liq_sig else "#64748b"
                    
                    reasons_html = "".join([f"<li style='color:#cbd5e1;font-size:0.85rem;'>{r}</li>" for r in liq.get("reasons", [])])
                    
                    st.markdown(
                        f"""
                        <div class="detail-card">
                            <div class="detail-label">💧 Liquidity Status</div>
                            <div class="detail-value" style="color:{liq_color};">{liq_sig}</div>
                            <div style="margin-top:10px;">
                                <ul style="padding-left:15px; margin:0;">
                                    {reasons_html}
                                </ul>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Regime Card
                with d2:
                    regime_val = regime
                    if "high volatility" in regime.lower():
                        reg_color = "#f43f5e" # Red
                        reg_icon = "⚠️"
                    elif "choppy" in regime.lower():
                        reg_color = "#f59e0b" # Orange
                        reg_icon = "🌊"
                    else:
                        reg_color = "#10b981" # Green
                        reg_icon = "✅"
                        
                    st.markdown(
                        f"""
                        <div class="detail-card">
                            <div class="detail-label">🔥 Market Regime</div>
                            <div class="detail-value" style="color:{reg_color};">{reg_icon} {regime_val}</div>
                            <div class="detail-sub" style="color:#94a3b8;">
                                Measures current price behavior stability.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # --- Row 2: Returns Calculator (Computed on fly) ---
                st.markdown("#### 🚀 Performance Snapshot")
                
                # Calculate simple returns if data exists
                try:
                    curr_close = df_daily['Close'].iloc[-1]
                    
                    ret_1d = (curr_close - df_daily['Close'].iloc[-2]) / df_daily['Close'].iloc[-2] * 100 if len(df_daily) >= 2 else 0
                    ret_1w = (curr_close - df_daily['Close'].iloc[-6]) / df_daily['Close'].iloc[-6] * 100 if len(df_daily) >= 6 else 0
                    ret_1m = (curr_close - df_daily['Close'].iloc[-21]) / df_daily['Close'].iloc[-21] * 100 if len(df_daily) >= 21 else 0
                except:
                    ret_1d, ret_1w, ret_1m = 0, 0, 0

                p1, p2, p3 = st.columns(3)
                p1.metric("1 Day Change", f"{ret_1d:.2f}%", delta=f"{ret_1d:.2f}%")
                p2.metric("1 Week Change", f"{ret_1w:.2f}%", delta=f"{ret_1w:.2f}%")
                p3.metric("1 Month Change", f"{ret_1m:.2f}%", delta=f"{ret_1m:.2f}%")

                st.markdown("---")

                # --- Row 3: Data Explorers ---
                st.markdown("#### 💾 Raw Data Inspector")
                
                with st.expander("📊 View Recent Daily Data (Last 15 Sessions)", expanded=True):
                    st.dataframe(
                        df_daily.tail(15).style.background_gradient(subset=['Close', 'Volume'], cmap="Blues"),
                        width='stretch'
                    )

                if df_intraday is not None and not df_intraday.empty:
                    with st.expander("⏰ View Intraday Data (Last 50 Ticks)", expanded=False):
                        st.dataframe(
                            df_intraday.tail(50), 
                            width='stretch'
                        )

    else:
        st.info("Click **🚀 START ANALYSIS** in the left sidebar to run the full analysis.")