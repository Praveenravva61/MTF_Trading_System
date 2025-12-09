# MTF Trading System - Professional Streamlit Application

A comprehensive Multi-Timeframe Trading Analysis platform built with Streamlit, providing real-time technical analysis, news sentiment, fundamental metrics, and swing trade setups for Indian stocks.

<<<<<<< HEAD
=======
# App visuals:

<img width="1900" height="856" alt="image" src="https://github.com/user-attachments/assets/260b760c-4dc2-431e-8867-b20cd604565f" />
<img width="1875" height="884" alt="image" src="https://github.com/user-attachments/assets/bb8cf5e0-4f87-4e74-a33f-0d747c3dc3f3" />
<img width="1918" height="917" alt="image" src="https://github.com/user-attachments/assets/c19d60a4-b047-4071-aac6-0c9b6f4087ff" />
<img width="1875" height="884" alt="image" src="https://github.com/user-attachments/assets/f2408cd6-e0dd-4a18-b79c-9400512bbbc3" />
<img width="1889" height="904" alt="image" src="https://github.com/user-attachments/assets/499b82ab-8dd8-47de-bc12-bd6fe73bf38a" />
<img width="1535" height="770" alt="image" src="https://github.com/user-attachments/assets/2a7c5615-1b0e-4bc6-b249-c690a14de5fc" />
<img width="1535" height="840" alt="image" src="https://github.com/user-attachments/assets/b04430e6-6c5e-4b66-8621-5524c54447b6" />
<img width="1853" height="853" alt="image" src="https://github.com/user-attachments/assets/385609c3-c585-44a6-a23b-30eafb83bb81" />
<img width="1920" height="867" alt="image" src="https://github.com/user-attachments/assets/47901a7b-a297-461a-9997-ee27aef4f671" />









>>>>>>> 65596a490f88b7354b84d34d5a52877ece134d6f
## 🌟 Features

### 📊 Technical Analysis
- **Multi-timeframe trend analysis** (Daily, Hourly, 15min, 5min)
- **8 Technical indicators**: SMA, EMA, MACD, RSI, Bollinger Bands, Stochastic, OBV, ADX
- **Aggregated signals** with confidence scoring
- **Support & Resistance levels** using DBSCAN clustering

### 📈 Visual Charts
- Interactive candlestick charts with volume
- Intraday price movements (1-minute data)
- Technical indicator overlays
- Volume analysis with moving averages
- Multi-timeframe trend visualization

### 📰 News Sentiment Analysis
- Real-time news fetching using Google Gemini AI
- Sentiment scoring (Bullish/Bearish/Neutral)
- Article aggregation from multiple sources
- Sentiment strength gauge

### 💼 Fundamental Analysis
- PE Ratio, PB Ratio, ROE, ROA
- Profit Margin, Debt-to-Equity
- Current Ratio, Quick Ratio
- Market Cap analysis
- Fundamental strength scoring (0-100%)

### 🎯 Swing Trade Setup
- Entry, Stop Loss, and Target prices
- Risk-Reward ratio calculation
- Pattern detection (Bullish Engulfing, Hammer)
- Breakout and retest identification
- Pullback analysis

### 🔍 Advanced Features
- **Liquidity Scanner**: Detects thin liquidity and slippage risks
- **Market Regime Detection**: Identifies high volatility and choppy markets
- **Safety Warnings**: Alerts for gap risks and extreme conditions
- **Confidence Scoring**: AI-powered signal confidence (20-95%)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the repository**
```bash
cd mtf_trading_system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up Google API Key (for news analysis)**

Create a `.streamlit/secrets.toml` file:
```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

Or set as environment variable:
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the application**
Open your browser and navigate to:
```
http://localhost:8501
```

## 📖 Usage Guide

### Stock Selection
1. Use the sidebar to select from popular Indian stocks
2. Or enter a custom ticker symbol (e.g., `RELIANCE.NS`, `TCS.NS`)
3. Click "🚀 Analyze Stock" button

### Understanding the Dashboard

#### 📊 Charts Tab
- **Daily Candlestick Chart**: Shows price action with support/resistance levels
- **Intraday Chart**: Real-time price movements for the current trading day
- **Volume Analysis**: Volume bars with 20-day moving average

#### 📈 Technical Analysis Tab
- **Indicator Signals**: View all 8 technical indicators at a glance
- **Multi-timeframe Trends**: Daily, Hourly, and Lower timeframe analysis
- **RSI, MACD, ADX**: Key momentum indicators

#### 🎯 Swing Trade Tab
- **Entry Price**: Suggested entry point
- **Stop Loss**: Risk management level
- **Target**: Profit target based on 2:1 risk-reward
- **Setup Score**: Confidence in the swing trade setup (0-100)
- **Key Levels**: Nearest support and resistance zones

#### 📰 News Tab
- **Sentiment Overview**: Bullish/Bearish/Neutral classification
- **Sentiment Strength**: Gauge showing conviction level
- **Recent Articles**: Latest news from multiple sources

#### 💼 Fundamentals Tab
- **Fundamental Score**: Overall strength (0-100%)
- **Key Metrics**: PE, PB, ROE, ROA, Profit Margin, etc.
- **Verdict**: Strong/Average/Weak classification

#### 🔍 Detailed Metrics Tab
- **Liquidity Analysis**: Volume strength and stability
- **Market Regime**: Volatility and trend strength
- **Raw Data Tables**: Recent price and volume data

### Signal Interpretation

#### 🟢 BUY Signal
- Multiple bullish indicators aligned
- Strong uptrend across timeframes
- Positive news sentiment
- Good fundamental strength
- **Action**: Consider entering long position

#### 🔴 SELL Signal
- Multiple bearish indicators aligned
- Downtrend across timeframes
- Negative news sentiment
- **Action**: Consider exiting or shorting

#### ⚪ HOLD Signal
- Mixed or neutral indicators
- Conflicting timeframe signals
- Uncertain market conditions
- **Action**: Wait for clearer setup

### Confidence Levels
- **70-95%**: High confidence - Strong signal
- **50-69%**: Medium confidence - Moderate signal
- **20-49%**: Low confidence - Weak signal

## ⚠️ Risk Warnings

The system provides automatic warnings for:
- **Thin Liquidity**: Low trading volume, high slippage risk
- **Extreme Volatility**: High ATR, increased stop-loss hit probability
- **Gap Risk**: Large price gaps indicating unstable conditions

## 🛠️ Project Structure

```
mtf_trading_system/
├── app.py                          # Main Streamlit application
├── modules/
│   ├── data_fetcher.py            # Yahoo Finance data fetching
│   ├── technical_analysis.py      # TA indicators and signals
│   ├── support_resistance.py      # S/R level detection
│   ├── news_analysis.py           # News sentiment with Gemini
│   ├── fundamentals.py            # Fundamental analysis
│   ├── mtf_engine.py              # Multi-timeframe aggregation
│   ├── swing_trading.py           # Swing trade setup
│   ├── market_regime.py           # Volatility detection
│   ├── liquidity.py               # Liquidity scanner
│   └── master_report.py           # Master orchestrator
├── utils/
│   └── visualizations.py          # Plotly chart functions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 📊 Supported Stocks

The application includes 30+ popular Indian stocks:
- Nifty 50 constituents
- Banking stocks (HDFC, ICICI, SBI, PNB)
- IT stocks (TCS, Infosys, Wipro, HCL)
- Auto stocks (Maruti, Tata Motors, Bajaj)
- Energy stocks (Reliance, NTPC, Power Grid)
- And many more...

You can also analyze any NSE/BSE listed stock by entering its ticker symbol.

## 🔧 Customization

### Adding New Stocks
Edit the `POPULAR_STOCKS` dictionary in `app.py`:
```python
POPULAR_STOCKS = {
    "Your Stock Name": "TICKER.NS",
    ...
}
```

### Adjusting Indicator Parameters
Modify parameters in `modules/technical_analysis.py`:
```python
def rsi(series, period=14):  # Change period here
def macd(series, fast=12, slow=26, signal=9):  # Adjust MACD params
```

### Changing Risk-Reward Ratio
In `modules/swing_trading.py`:
```python
def swing_trade(df1, symbol, rr=2.0):  # Change rr value
```

## 📝 Technical Details

### Data Sources
- **Price Data**: Yahoo Finance API
- **News**: Google Search API via Gemini AI
- **Fundamentals**: Yahoo Finance company info

### Indicators Used
1. **SMA (50, 200)**: Trend identification
2. **EMA (12, 26)**: Short-term momentum
3. **MACD (12, 26, 9)**: Trend and momentum
4. **RSI (14)**: Overbought/oversold conditions
5. **Bollinger Bands (20, 2)**: Volatility and extremes
6. **Stochastic (14, 3)**: Momentum oscillator
7. **OBV**: Volume-price relationship
8. **ADX (14)**: Trend strength

### Signal Aggregation
Signals are weighted and aggregated:
- SMA: 2x weight
- EMA: 2x weight
- MACD: 3x weight
- RSI: 1.5x weight
- Bollinger: 1.5x weight
- Stochastic: 1x weight
- OBV: 1x weight
- ADX: 2x weight

Final score: -1 (Strong Sell) to +1 (Strong Buy)

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs
- Suggest features
- Submit improvements
- Share feedback

## ⚠️ Disclaimer

**IMPORTANT**: This application is for **educational and informational purposes only**. 

- **NOT financial advice**: Do not use this as the sole basis for investment decisions
- **No guarantees**: Past performance does not indicate future results
- **Risk warning**: Trading involves substantial risk of loss
- **Do your research**: Always conduct your own analysis
- **Consult professionals**: Seek advice from qualified financial advisors

The developers are not responsible for any financial losses incurred from using this tool.

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- **yfinance**: For providing free stock data
- **Streamlit**: For the amazing web framework
- **Plotly**: For interactive visualizations
- **Google Gemini**: For AI-powered news analysis
- **scikit-learn**: For clustering algorithms

---

**Built with ❤️ for traders and developers**

<<<<<<< HEAD
For questions or support, please refer to the documentation or create an issue in the repository.
=======
For questions or support, please refer to the documentation or create an issue in the repository.
>>>>>>> 65596a490f88b7354b84d34d5a52877ece134d6f
