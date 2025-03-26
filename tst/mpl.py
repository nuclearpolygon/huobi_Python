import np_mplchart as mc
import pandas as pd
import sqlalchemy

engine = sqlalchemy.create_engine("sqlite:///./financial_data.db", execution_options={"sqlite_raw_colnames": True})

df = pd.read_sql_table('btcusdt_1min', con=engine)


class Chart(mc.SliderChart):
    def __init__(self, symbol, interval):
        super().__init__()
        # db.fetch_data(symbol, interval)
        df = pd.read_sql_table(f'{symbol}_{interval}', con=engine)
        mc.set_theme(self, 'dark')
        self.watermark = ''
        self.date = 'Date'
        self.Open = 'Open'
        self.high = 'High'
        self.low = 'Low'
        self.close = 'Close'
        self.volume = 'Volume'
        self.list_ma = (1, 2, 4, 8, 16)
        self.set_data(df)
        self.figure.set_figheight(4)
        print(self.figure.axes)

    def qtWidget(self):
        return self.figure.canvas

