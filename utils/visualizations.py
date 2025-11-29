"""Visualization utilities using Plotly"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_candlestick_chart(df, title="Stock Price", show_volume=True, support_resistance=None):
    """Create interactive candlestick chart with volume."""
    
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(title, 'Volume')
        )
    else:
        fig = go.Figure()
    
    # Candlestick
    candlestick = go.Candlestick(
        x=df['Date'] if 'Date' in df.columns else df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
        
        # Volume bars
        colors = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350' 
                  for i in range(len(df))]
        
        volume = go.Bar(
            x=df['Date'] if 'Date' in df.columns else df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        )
        fig.add_trace(volume, row=2, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add support and resistance levels
    if support_resistance:
        x_range = df['Date'] if 'Date' in df.columns else df.index
        
        for level in support_resistance.get('nearest_resistances', []):
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="red", 
                annotation_text=f"R: ₹{level}",
                annotation_position="right",
                row=1, col=1
            )
        
        for level in support_resistance.get('nearest_supports', []):
            fig.add_hline(
                y=level, 
                line_dash="dash", 
                line_color="green", 
                annotation_text=f"S: ₹{level}",
                annotation_position="right",
                row=1, col=1
            )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark',
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=2 if show_volume else 1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def create_indicator_chart(df, indicator_name, indicator_data, signal_lines=None):
    """Create chart for technical indicators."""
    
    fig = go.Figure()
    
    x_data = df['Date'] if 'Date' in df.columns else df.index
    
    # Main indicator line
    fig.add_trace(go.Scatter(
        x=x_data,
        y=indicator_data,
        mode='lines',
        name=indicator_name,
        line=dict(color='#2196F3', width=2)
    ))
    
    # Signal lines (e.g., RSI overbought/oversold)
    if signal_lines:
        for line_val, line_name, line_color in signal_lines:
            fig.add_hline(
                y=line_val,
                line_dash="dash",
                line_color=line_color,
                annotation_text=line_name,
                annotation_position="right"
            )
    
    fig.update_layout(
        title=indicator_name,
        xaxis_title="Date",
        yaxis_title=indicator_name,
        height=300,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig


def create_macd_chart(df, macd_line, signal_line, histogram):
    """Create MACD indicator chart."""
    
    fig = go.Figure()
    
    x_data = df['Date'] if 'Date' in df.columns else df.index
    
    # MACD Line
    fig.add_trace(go.Scatter(
        x=x_data,
        y=macd_line,
        mode='lines',
        name='MACD',
        line=dict(color='#2196F3', width=2)
    ))
    
    # Signal Line
    fig.add_trace(go.Scatter(
        x=x_data,
        y=signal_line,
        mode='lines',
        name='Signal',
        line=dict(color='#FF9800', width=2)
    ))
    
    # Histogram
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in histogram]
    fig.add_trace(go.Bar(
        x=x_data,
        y=histogram,
        name='Histogram',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='MACD Indicator',
        xaxis_title="Date",
        yaxis_title="MACD",
        height=300,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig


def create_intraday_chart(df_intraday, title="Intraday Price Movement"):
    """Create intraday line chart."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_intraday.index,
        y=df_intraday['Close'],
        mode='lines',
        name='Price',
        line=dict(color='#2196F3', width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price (₹)",
        height=400,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig


def create_gauge_chart(value, title, max_value=100):
    """Create gauge chart for confidence/score."""
    
    if value >= 70:
        color = "#26a69a"
    elif value >= 40:
        color = "#FFA726"
    else:
        color = "#ef5350"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "rgba(239, 83, 80, 0.2)"},
                {'range': [33, 66], 'color': "rgba(255, 167, 38, 0.2)"},
                {'range': [66, 100], 'color': "rgba(38, 166, 154, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        template='plotly_dark'
    )
    
    return fig


def create_mtf_trend_chart(analyses):
    """Create multi-timeframe trend visualization."""
    
    timeframes = ['Daily', '1H', '15min', '5min']
    trends = []
    colors = []
    
    for tf in timeframes:
        if tf in analyses:
            trend = analyses[tf].get('ema_trend', 'neutral')
            trends.append(trend.upper())
            
            if trend == 'bull':
                colors.append('#26a69a')
            elif trend == 'bear':
                colors.append('#ef5350')
            else:
                colors.append('#9E9E9E')
        else:
            trends.append('N/A')
            colors.append('#9E9E9E')
    
    fig = go.Figure(data=[
        go.Bar(
            x=timeframes,
            y=[1, 1, 1, 1],
            text=trends,
            textposition='inside',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Trend: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Multi-Timeframe Trend Analysis',
        xaxis_title="Timeframe",
        yaxis_visible=False,
        height=250,
        template='plotly_dark',
        showlegend=False
    )
    
    return fig


def create_volume_analysis_chart(df):
    """Create volume analysis with moving average."""
    
    fig = go.Figure()
    
    x_data = df['Date'] if 'Date' in df.columns else df.index
    
    # Volume bars
    colors = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350' 
              for i in range(len(df))]
    
    fig.add_trace(go.Bar(
        x=x_data,
        y=df['Volume'],
        name='Volume',
        marker_color=colors
    ))
    
    # Volume MA
    vol_ma = df['Volume'].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=x_data,
        y=vol_ma,
        mode='lines',
        name='20-day Avg',
        line=dict(color='#FFA726', width=2)
    ))
    
    fig.update_layout(
        title='Volume Analysis',
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    return fig