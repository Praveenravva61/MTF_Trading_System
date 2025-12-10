# 📈 MTF Trading System  
### AI-Powered Multi-Timeframe Stock Forecasting & Market Intelligence Platform

The **MTF Trading System** is an end-to-end, production-grade stock market analysis platform designed to deliver **high-accuracy forecasts**, **multi-timeframe technical analysis**, **sentiment intelligence**, and **fundamental scoring**—all inside an interactive **Streamlit dashboard**.

Built with **Deep Learning, Time-Series Modeling, and Modern Market Analytics**, this project demonstrates strong capabilities in:

- Machine Learning & Deep Learning  
- Financial data engineering  
- Large-scale technical analysis  
- AI-driven sentiment modeling  
- Modular Python architecture  
- Streamlit UI design  
- Real-world software engineering  

This project is engineered to be **resume-ready**, **interview-ready**, and **industry-ready**.

---

# 🌟 Why This Project Stands Out

✔ Multi-step deep learning forecasting (60-day horizon)  
✔ Multi-timeframe analysis (Daily, Hourly, 15m, 5m)  
✔ Advanced feature engineering  
✔ Market regime detection  
✔ Automated news sentiment via Google Gemini  
✔ Fundamental scoring engine  
✔ Interactive visual dashboards  
✔ Modular, production-ready architecture  

---

# 🖼 Application Visuals

<img width="1900" height="856" alt="image" src="https://github.com/user-attachments/assets/260b760c-4dc2-431e-8867-b20cd604565f" />
<img width="1583" height="846" alt="image" src="https://github.com/user-attachments/assets/e21814c7-b523-43f2-9887-c32312b9e738" />
<img width="1564" height="831" alt="image" src="https://github.com/user-attachments/assets/328e2282-a67c-432f-badf-982dab400a98" />
<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/c19d60a4-b047-4071-aac6-0c9b6f4087ff" />
<img width="1875" height="884" alt="image" src="https://github.com/user-attachments/assets/f2408cd6-e0dd-4a18-b79c-9400512bbbc3" />
<img width="1889" height="904" alt="image" src="https://github.com/user-attachments/assets/499b82ab-8dd8-47de-bc12-bd6fe73bf38a" />
<img width="1535" height="770" alt="image" src="https://github.com/user-attachments/assets/2a7c5615-1b0e-4bc6-b249-c690a14de5fc" />
<img width="1535" height="840" alt="image" src="https://github.com/user-attachments/assets/b04430e6-6c5e-4b66-8621-5524c54447b6" />
<img width="1853" height="853" alt="image" src="https://github.com/user-attachments/assets/385609c3-c585-44a6-a23b-30eafb83bb81" />
<img width="1920" height="867" alt="image" src="https://github.com/user-attachments/assets/47901a7b-a297-461a-9997-ee27aef4f671" />

---

# 🚀 Core System Features

## 🔮 1. Deep Learning Forecasting Engine (60-Day Horizon)

✔ Conv1D — Local price pattern detection  
✔ Bi-LSTM — Sequence memory  
✔ Multi-Head Attention — Identifying important historical zones  
✔ Residual Connections — Stable gradient flow  
✔ GELU Activation — Transformer-like performance  

Outputs:  
- 60-step log return forecast  
- Reconstructed price curve  
- Confidence scoring  

---

## 📊 2. Technical Analysis (Multi-Timeframe)

Includes:  
- SMA, EMA, MACD, RSI, ADX  
- Bollinger Bands, OBV, Stochastic  
- DBSCAN S/R clustering  
- Trend strength + signal confidence score  

---

## 📰 3. News Sentiment Engine

✔ Fetches stock news using Google Gemini  
✔ Generates sentiment score (Bullish/Bearish/Neutral)  
✔ Computes sentiment strength  
✔ Summaries and signal mapping  

---

## 💼 4. Fundamental Analysis Engine

✔ PE, PB, ROA, ROE  
✔ Profit Margins  
✔ Debt Ratios  
✔ Market Cap  
✔ Fundamental strength score (0–100%)  

---

## 🎯 5. Swing Trading Assistant

✔ Entry, Stop Loss & Targets  
✔ Risk–Reward (RRR) calculation  
✔ Pattern detection (Engulfing, Hammer, Breakout-Retest)  
✔ Pullback validation  
✔ Swing Setup Confidence Score  

---

## 🔍 6. Market Regime & Liquidity Scanner

✔ Volatility classification  
✔ Liquidity strength  
✔ Gap-risk detector  
✔ Choppiness index  
✔ Trend vs Sideways regime detection  

---

## 📊 7. Interactive Streamlit Dashboard

- Forecast visualization  
- Technical indicators  
- Market regime insights  
- News & sentiment  
- Fundamentals  
- Swing trade setups  
- Detailed metrics  

---

# 🏗 Project Structure

```
MTF_TRADING_SYSTEM/
│
├── app.py                     # Streamlit UI
├── requirements.txt           # Dependency list
├── README.md                  # Documentation
│
├── models/                    # Saved ML models & processed datasets
│   ├── SYMBOL_model.keras
│   └── SYMBOL_data.pkl
│
├── Images/                    # Dashboard screenshots
│
├── modules/                   # Core analysis engines
│   ├── data_fetcher.py
│   ├── forecasting.py
│   ├── fundamentals.py
│   ├── liquidity.py
│   ├── market_regime.py
│   ├── master_report.py
│   ├── mtf_engine.py
│   ├── news_analysis.py
│   ├── support_resistance.py
│   ├── swing_trading.py
│   └── technical_analysis.py
│
└── utils/
    ├── visualizations.py
    └── __init__.py
```

---

# 🧠 Feature Engineering

| Feature | Description |
|--------|-------------|
| Log_Returns | Trend driver |
| High_Low_Ratio | Volatility |
| Close_Open_Ratio | Directional bias |
| Dist_SMA_10/20/50 | Trend deviation |
| MACD_Line | Momentum acceleration |
| RSI | Overbought/Oversold |
| ATR_Pct | Volatility intensity |
| BB_Width | Breakout probability |
| BB_Pos | Band location |
| Vol_Ratio | Institutional volume |

Stationary, normalized, volatility-adjusted features provide **stable model training**.

---

# 🤖 Forecasting Architecture

```
Input → Conv1D → LN → GELU → Dropout  
      → Bi-LSTM → LN  
      → Multi-Head Attention + Residual  
      → Global Avg Pool  
      → Dense → Dropout  
      → Output(60)
```

Optimized for:  
✔ Multi-horizon prediction  
✔ Long-range temporal learning  
✔ Low overfitting  
✔ Fast inference  

---

# 🛠 Installation

```bash
git clone https://github.com/your-username/MTF_TRADING_SYSTEM.git
cd MTF_TRADING_SYSTEM
pip install -r requirements.txt
```

---

# ▶️ Running the Application

```bash
streamlit run app.py
```

---

# 📦 Dependencies (Key Libraries)

- tensorflow, keras  
- scikit-learn  
- xgboost  
- statsmodels  
- pmdarima  
- ta  
- yfinance  
- plotly, seaborn  
- streamlit  
- google-generativeai  

---

# 🎓 Skills Demonstrated (Recruiter Focus)

✔ Deep Learning (LSTM, Attention, CNN)  
✔ Time-Series Forecasting  
✔ Financial Feature Engineering  
✔ NLP Sentiment Analysis  
✔ End-to-End ML System Architecture  
✔ Modular Python Development  
✔ Streamlit UI Design  
✔ Data Engineering & Visualization  
✔ Real-world trading analytics  

This project clearly communicates your **AI + Finance + Full-Stack ML** capabilities.

---

# 📄 License  
MIT License  

# 🙌 Acknowledgments  
yfinance, Streamlit, Plotly, Google Gemini, scikit-learn

---

# ❤️ Built with passion for trading & AI  
