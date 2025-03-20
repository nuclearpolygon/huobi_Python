import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pandas_datareader.data import DataReader
import yfinance as yf

from datetime import datetime
from pathlib import Path

tech_list = ['BTC-USD', 'XRP-USD']

end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
BTCUSD: pd.DataFrame
XRPUSD: pd.DataFrame
company_name = ["Bitcoin", "Ripple"]

for i, stock in enumerate(tech_list):
    stock_var = stock.replace('-', '')
    f_name = Path(f'{company_name[i]}.pkl')
    if not f_name.exists():
        globals()[stock_var] = yf.download(stock, period='max', interval='1m')
    else:
        globals()[stock_var] = pd.read_pickle(f_name.__str__())

company_list = [BTCUSD, XRPUSD]
for i, company in enumerate(company_list):
    f_name = Path(f'{company_name[i]}.pkl')
    if not f_name.exists():
        company.to_pickle(f_name.__str__())


ma_day = [10, 20, 50]
for ma in ma_day:
    for i, company in enumerate(company_list):
        column_name = f"MA{ma}"
        company[column_name] = company['Close'].rolling(ma).mean()
        company['Return'] = company['Close'][tech_list[i]].pct_change()


# df = pd.concat(company_list, axis=0)

plt.figure(figsize=(15, 10))
for i, symbol in enumerate(company_list, 1):
    ax1 = plt.subplot(2, 1, i)
    plt.title(company_name[i - 1])
    ax1.plot(symbol.index, symbol['Close'], linewidth=2, label='Price')
    ax2 = ax1.twinx()
    ax2.plot(symbol.index, symbol['Volume'], linewidth=2, label='Volume', color='red')
    ax2.grid(False)
    ax2.set_ylim(0, float(symbol['Volume'].max()) * 3)
    ax3 = ax1.twinx()
    ax3.plot(symbol.index, symbol['Return'], linewidth=1, label='Return', linestyle='--')
    ax3.grid(False)
    for ma in ma_day:
        ax1.plot(symbol.index, symbol[f'MA{ma}'], linewidth=2, label=f'MA {ma}')

plt.tight_layout()
plt.show()


