from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *
from huobi.utils import *
import pandas as pd
import sqlite3
import time

market_client = MarketClient()
interval = CandlestickInterval.HOUR4
symbol = "ethusdt"
size = 1000
list_obj = market_client.get_candlestick(symbol, interval, size)
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
conn = sqlite3.connect('financial_data.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS stock_data (
    Date TEXT PRIMARY KEY,
    Open REAL,
    High REAL,
    Low REAL,
    Close REAL,
    Volume INTEGER
)
''')
df.to_sql('stock_data', conn, if_exists='replace', index=False)
conn.commit()
conn.close()