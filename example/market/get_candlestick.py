from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *
from huobi.utils import *
import pandas as pd
import mplfinance as mpf
from datetime import datetime
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import mpld3

market_client = MarketClient()
interval = CandlestickInterval.HOUR4
symbol = "ethusdt"
size = 1000
list_obj = market_client.get_candlestick(symbol, interval, size)
# LogInfo.output("---- {interval} candlestick for {symbol} ----".format(interval=interval, symbol=symbol))
# LogInfo.output_list(list_obj)
# date = datetime.now().date().strftime('%Y-%m-%d %H:%M')
date = pd.to_datetime(time.time(), unit='s')

data = {
    'Date': pd.date_range(end=date, periods=size, freq='4h'),
    'Open': [],
    'High': [],
    'Low': [],
    'Close': [],
    'Volume': []
}
kline: Candlestick
for kline in list_obj:
    data['Open'].append(kline.open)
    data['High'].append(kline.high)
    data['Low'].append(kline.low)
    data['Close'].append(kline.close)
    data['Volume'].append(kline.vol)
df = pd.DataFrame(data)

# Create the candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])

# Add volume as a bar chart
fig.add_trace(go.Bar(
    x=df['Date'],
    y=df['Volume'],
    name='Volume',
    marker_color='rgba(100, 100, 100, 0.6)',
    yaxis='y2'
))

# Update layout for better visualization
fig.update_layout(
    title='Candlestick Chart with Volume',
    xaxis_title='Date',
    yaxis_title='Price',
    yaxis2=dict(title='Volume', overlaying='y', side='right'),
    xaxis_rangeslider_visible=True
)

# Save the plot as an HTML file
fig.write_html("candlestick_chart_plotly.html")

print("Plot saved as 'candlestick_chart_plotly.html'. Open this file in your browser.")
