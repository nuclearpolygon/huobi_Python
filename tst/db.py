from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *
from huobi.utils import *
import pandas as pd
import sqlite3
import time
import datetime
import logging, sys
from PySide6.QtCore import QDateTime
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)



def time_mod(time, delta, epoch=None):
    if epoch is None:
        epoch = datetime.datetime(1970, 1, 1, tzinfo=time.tzinfo)
    return (time - epoch) % delta

def time_floor(time, delta, epoch=None):
    mod = time_mod(time, delta, epoch)
    return time - mod


Intervals = {
    CandlestickInterval.MIN1: {'minutes': 1},
    CandlestickInterval.MIN5: {'minutes': 5},
    CandlestickInterval.MIN15: {'minutes': 15},
    CandlestickInterval.MIN30: {'minutes': 30},
    CandlestickInterval.MIN60: {'hours': 1},
    CandlestickInterval.HOUR4: {'hours': 4},
    CandlestickInterval.DAY1: {'days': 1},
    CandlestickInterval.WEEK1: {'days': 7},
    CandlestickInterval.MON1: {'months': 1},
}
DatetimeIntervals = {
    CandlestickInterval.MIN1: datetime.timedelta(0, 0, 0, minutes=1),
    CandlestickInterval.MIN5: datetime.timedelta(0, 0, 0, minutes=5),
    CandlestickInterval.MIN15: datetime.timedelta(0, 0, 0, minutes=15),
    CandlestickInterval.MIN30: datetime.timedelta(0, 0, 0, minutes=30),
    CandlestickInterval.MIN60: datetime.timedelta(0, 0, 0, hours=1),
    CandlestickInterval.HOUR4: datetime.timedelta(0, 0, 0, hours=4),
    CandlestickInterval.DAY1: datetime.timedelta(0, 0, 1),
    CandlestickInterval.WEEK1: datetime.timedelta(0, 0, 7),
    CandlestickInterval.MON1: datetime.timedelta(0, 1, 0),
}

def round_time(_interval):
    t = datetime.datetime.fromtimestamp(time.time())
    delta = datetime.timedelta(**Intervals[_interval])
    epoch = datetime.datetime(1970, 1, 1, tzinfo=t.tzinfo)
    now = datetime.datetime.fromtimestamp((time_floor(t, delta) - epoch).total_seconds())
    return now


market_client = MarketClient(init_log=False)
def fetch_data(symbol, interval, size=2000):
    if size < 1:
        return
    size = min(size, 2000)
    list_obj = market_client.get_candlestick(symbol, interval, size)
    table_name = f'{symbol}_{interval}'
    data = []
    kline: Candlestick
    # log.info('START FETCHING DATA')
    with sqlite3.connect('financial_data.db') as conn:
        cursor = conn.cursor()
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            Date DATETIME PRIMARY KEY,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER
        )
        ''')
        for kline in list_obj:
            data.append(kline.asTuple())
            # print(kline.asTuple())
            # date = pd.to_datetime(kline.id, unit='s')
            # print(date)
            # print('=============')
        conn.executemany(f'INSERT INTO {table_name} (Date, Open, High, Low, Close, Volume) '
                     f'VALUES ({("?, "*6)[:-2]}) '
                     f'ON CONFLICT(Date) DO NOTHING;', data)
        conn.commit()
    # log.info(f'table {table_name} fetched')


# fetch_data('btcusdt', '5min')