from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *
from huobi.utils import *
import pandas as pd
import sqlite3
import time
import datetime


def time_mod(time, delta, epoch=None):
    if epoch is None:
        epoch = datetime.datetime(1970, 1, 1, tzinfo=time.tzinfo)
    return (time - epoch) % delta

def time_floor(time, delta, epoch=None):
    mod = time_mod(time, delta, epoch)
    return time - mod

Intervals = {
    CandlestickInterval.HOUR4: {'hours': 4}
}

def round_time(_interval):
    t = datetime.datetime.fromtimestamp(time.time())
    delta = datetime.timedelta(**Intervals[_interval])
    epoch = datetime.datetime(1970, 1, 1, tzinfo=t.tzinfo)
    now = (time_floor(t, delta) - epoch).total_seconds()
    return now



market_client = MarketClient()
interval = CandlestickInterval.HOUR4

symbol = "ethusdt"
size = 2000
list_obj = market_client.get_candlestick(symbol, interval, size)
date = pd.to_datetime(round_time(interval), unit='s')

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
    Date DATETIME PRIMARY KEY,
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