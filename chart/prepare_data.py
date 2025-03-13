import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import numpy as np

from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.exception.huobi_api_exception import HuobiApiException
from huobi.model.market import CandlestickEvent
from huobi.model.market import CandlestickReq


def callback(candlestick_req: CandlestickEvent):
    print(candlestick_req.tick.print_object())
    # np.append(y_data, candlestick_req.tick.close)
    # np.append(x_data, candlestick_req.tick.id)
    # # plt.ion()
    #
    # # Update the plot data
    # graph.clear()
    # graph.plot(x_data, y_data)
    #
    # # Adjust the plot limits if necessary
    # graph.relim()
    # graph.autoscale_view()

    # Redraw the plot
    # plt.draw()
    # plt.ioff()


def error(e: 'HuobiApiException'):
    print(e.error_code + e.error_message)

start = datetime(2020, 1, 1)
interval = timedelta(minutes=1)
end = datetime(2026, 1, 1)
market_client = MarketClient()
candles = market_client.get_candlestick('btcusdt', CandlestickInterval.MIN1, 2000)
# market_client.sub_candlestick("btcusdt", CandlestickInterval.MIN1, callback, error)
data = np.array([c.close for c in candles])
deriv = np.gradient(data, 1)
print(data)
print(deriv)
colors = []
up = np.ma.masked_where(deriv < 0, data)
down = np.ma.masked_where(deriv >= 0, data)
x_axis = np.arange(len(data))
plt.plot(x_axis, up, c='green', linestyle='-', label='Data')
plt.plot(x_axis, down, c='red', linestyle='-', label='Data')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Python List Data')

# Add a legend
plt.legend()

# Show the plot
plt.show()