import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta

import numpy as np

from huobi.client.market import MarketClient
from huobi.constant import *
from huobi.exception.huobi_api_exception import HuobiApiException
from huobi.model.market import CandlestickEvent
from huobi.model.market import CandlestickReq

market_client = MarketClient()
candles = market_client.get_candlestick('btcusdt', CandlestickInterval.MIN1, 20)
data = np.array([c.close for c in candles])
x_axis = np.arange(len(data))
deriv = np.gradient(data, 1)
up = np.ma.masked_where(deriv <= 0, data)
down = np.ma.masked_where(deriv >= 0, data)

def callback(candlestick_req: CandlestickEvent):
    print(candlestick_req.tick.print_object())
    # np.append(y_data, candlestick_req.tick.close)
    np.append(data, candlestick_req.tick.close)
    _x_axis = np.arange(len(data))
    _deriv = np.gradient(data, 1)
    _up = np.ma.masked_where(_deriv <= 0, data)
    _down = np.ma.masked_where(_deriv >= 0, data)
    # plt.plot(x_axis, up, c='green', linestyle='-', label='Data')
    # plt.plot(x_axis, down, c='red', linestyle='-', label='Data')
    ax.clear()
    ax.plot(_x_axis, _up, c='green', linestyle='-', label='Data')
    ax.plot(_x_axis, _down, c='red', linestyle='-', label='Data')
    fig.canvas.draw_idle()
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
    plt.draw()
    # plt.ioff()



def error(e: 'HuobiApiException'):
    print(e.error_code + e.error_message)

market_client.sub_candlestick("btcusdt", CandlestickInterval.MIN1, callback, error)

fig = plt.figure()
ax = fig.add_subplot()
line_up, = ax.plot(x_axis, up, c='green', linestyle='-', label='Data')
line_down, = ax.plot(x_axis, down, c='red', linestyle='-', label='Data')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Plot of Python List Data')

# Add a legend
plt.legend()

# Show the plot
plt.show()