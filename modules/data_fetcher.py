"""Data fetching module for Yahoo Finance"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_yahoo_finance_history(ticker, period='max', start_date=None, end_date=None, interval='1d'):
    """
    Fetch historical stock data from Yahoo Finance and return a cleaned DataFrame.
    """
    try:
        stock = yf.Ticker(ticker)
        
        if start_date:
            df = stock.history(start=start_date, end=end_date, interval=interval)
        else:
            df = stock.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'])
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Adj Close' in df.columns:
            df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
        
        df = df.dropna(how='all', subset=[col for col in df.columns if col != 'Date'])
        
        cols_to_drop = ['Dividends', 'Stock Splits']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        df = df.sort_values('Date').reset_index(drop=True)
        
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        if 'Volume' in df.columns:
            df['Volume'] = df['Volume'].astype('Int64')
        
        return df
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()


def get_stock_info(ticker):
    """Get basic information about the stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        relevant_info = {
            'Name': info.get('longName', 'N/A'),
            'Symbol': info.get('symbol', ticker),
            'Currency': info.get('currency', 'N/A'),
            'Exchange': info.get('exchange', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Current Price': info.get('currentPrice', 'N/A'),
        }
        
        return relevant_info
    except Exception as e:
        print(f"Error fetching stock info: {str(e)}")
        return {}


def fetch_yahoo_intraday_1min(symbol: str, date: str) -> pd.DataFrame:
    """Fetch 1-minute intraday data for a specific date."""
    try:
        day = pd.to_datetime(date)
        start = day
        end = day + pd.Timedelta(days=1)

        df = yf.download(symbol, interval="1m", start=start, end=end, progress=False)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index, errors="coerce")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
        else:
            df.index = df.index.tz_convert("Asia/Kolkata")

        df = df.between_time("09:15", "15:30")
        df = df.reset_index(drop=False)

        datetime_col = df.columns[0]
        df["Time"] = df[datetime_col].dt.strftime("%H:%M")
        df = df.drop(columns=[datetime_col])

        if "Price" in df.columns:
            df = df.drop(columns=["Price"])

        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                df[col] = df[col].astype(float).round(2)

        df = df.set_index("Time")
        df.index.name = "Time"
        df.columns.name = None

        return df
    except Exception as e:
        print(f"Error fetching intraday data: {str(e)}")
        return pd.DataFrame()


def get_latest_intraday_data(symbol: str, days_back=5) -> pd.DataFrame:
    """Get latest available intraday data by searching last N days."""
    for i in range(days_back):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        try:
            df = fetch_yahoo_intraday_1min(symbol, date)
            if not df.empty:
                return df
        except:
            continue
    return pd.DataFrame()