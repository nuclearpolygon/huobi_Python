import sys
from pprint import pformat

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QSizePolicy
from PySide6.QtCharts import QCandlestickSeries, QCandlestickSet, QCandlestickModelMapper, QChart, QValueAxis, QChartView, QDateTimeAxis
from PySide6.QtCore import QDateTime, Qt, QObject, Slot, Property, Signal
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from PySide6.QtGui import QPalette, QColor, QFont

from ui.ui_mainwindow import Ui_Form
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy.sql import text

import db

engine = sqlalchemy.create_engine("sqlite:///./financial_data.db",
                                       execution_options={"sqlite_raw_colnames": True})


def round_time(now, _timedelta):
    return datetime.fromtimestamp(now.timestamp() - (now.timestamp() % _timedelta.total_seconds()))

def time_range(start: datetime, end: datetime, _timedelta: timedelta):
    while start + _timedelta != end - _timedelta:
        start += _timedelta
        yield start


class Backend(QObject):
    def __init__(self):
        super().__init__()
        self._first_value = 20  # Initial first value
        self._second_value = 80  # Initial second value

    # Signals to notify QML of property changes
    updateFirst = Signal(float)
    updateSecond = Signal(float)

    # Slot to update the first value
    @Slot(float)
    def updateFirstValue(self, value):
        self._first_value = value
        self.updateFirst.emit(value)

    # Slot to update the second value
    @Slot(float)
    def updateSecondValue(self, value):
        self._second_value = value
        self.updateSecond.emit(value)


class CandleChart(QChartView):
    def __init__(self, symbol, interval):
        self.table_name = f'{symbol}_{interval}'
        self.symbol = symbol
        self.interval = interval
        self.timedelta = db.DatetimeIntervals[interval]
        self.init_db()

        # Create a candlestick chart
        self.chart = QChart()
        self.chart.setTitle(f"{symbol}_{interval}")
        self.chart.setTitleBrush(QColor.fromString('white'))
        self.chart.legend().hide()
        self.chart.setBackgroundBrush(QColor(50, 50, 50, 255))
        # Create a chart view and add it to the layout
        super().__init__(self.chart)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        # Create a candlestick series
        self.series = QCandlestickSeries()
        self.series.setIncreasingColor(Qt.GlobalColor.green)
        self.series.setDecreasingColor(Qt.GlobalColor.red)
        self.series.setBodyOutlineVisible(False)

        # Fetch data from the SQLite database
        query = QSqlQuery(f"SELECT Date, Open, High, Low, Close FROM {symbol}_{interval} ORDER BY Date")
        prev_date = None
        while query.next():
            date = datetime.fromisoformat(query.value("Date"))
            if not prev_date:
                prev_date = date - self.timedelta
            if date - self.timedelta != prev_date:
                for gap_date in time_range(prev_date, date, self.timedelta):
                    candlestick_set = QCandlestickSet(
                        timestamp=gap_date.timestamp() * 1000
                    )
                    self.series.append(candlestick_set)
            prev_date = date
            open_price = query.value("Open")
            high_price = query.value("High")
            low_price = query.value("Low")
            close_price = query.value("Close")

            # Create a candlestick set and add it to the series
            candlestick_set = QCandlestickSet(
                open_price, high_price, low_price, close_price,  date.timestamp() * 1000
            )
            self.series.append(candlestick_set)

        # Add the series to the chart
        self.chart.addSeries(self.series)

        # Create axes
        self.chart.createDefaultAxes()
        self.axis_x = self.chart.axes(Qt.Orientation.Horizontal)[0]
        self.axis_x.setVisible(False)
        categories = self.axis_x.categories()
        start_id = 0
        end_id = -1
        start = QDateTime.fromString(categories[start_id], 'dd.MM.yyyy hh:mm')
        end = QDateTime.fromString(categories[end_id], 'dd.MM.yyyy hh:mm')

        self.date_axis = QDateTimeAxis()
        self.date_axis.setRange(start, end)
        self.axis_x.setRange(categories[start_id], categories[end_id])
        self.chart.addAxis(self.date_axis, Qt.AlignmentFlag.AlignBottom)

        # Format the y-axis
        self.axis_y = self.chart.axes(Qt.Orientation.Vertical)[0]
        self.axis_y.setTitleText("Price")
        self.axis_y.setTitleBrush(QColor.fromString('white'))
        self.axis_y.setLabelsColor(QColor.fromString('white'))
        self.date_axis.setLabelsColor(QColor.fromString('white'))

    def init_db(self):
        query = QSqlQuery(f"SELECT max(Date) as Date FROM {self.table_name}")
        query.next()
        date = query.value('Date')
        last_date = datetime.fromtimestamp(QDateTime.fromString(date, 'yyyy-MM-dd hh:mm:ss').toSecsSinceEpoch())
        now = datetime.now()
        now = round_time(now, self.timedelta)
        intervals_count = 0
        while last_date < now:
            now -= self.timedelta
            intervals_count += 1
        db.fetch_data(self.symbol, self.interval, intervals_count)

    @property
    def start(self):
        return self.date_axis.min().toString('yyyy-MM-dd hh:mm:ss')

    @start.setter
    def start(self, value):
        categories = self.axis_x.categories()
        start = int(value * (len(categories) - 1))
        self.axis_x.setMin(categories[start])
        start_date = QDateTime.fromString(categories[start], 'dd.MM.yyyy hh:mm')
        self.date_axis.setMin(start_date)
        self.get_y_bounds()

    @property
    def end(self):
        return self.date_axis.max().toString('yyyy-MM-dd hh:mm:ss')

    @end.setter
    def end(self, value):
        categories = self.axis_x.categories()
        end = int(value * (len(categories) - 1))
        self.axis_x.setMax(categories[end])
        end_date = QDateTime.fromString(categories[end], 'dd.MM.yyyy hh:mm')
        self.date_axis.setMax(end_date)
        self.get_y_bounds()

    def get_y_bounds(self):
        with engine.connect() as conn:
            q = text(
                f"SELECT max(High) FROM {self.table_name} "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _max = conn.execute(q, {'r0': self.start, 'r1': self.end}).scalar_one()
            q = text(
                f"SELECT min(Low) FROM {self.table_name} "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _min = conn.execute(q, {'r0': self.start, 'r1': self.end}).scalar_one()
            self.axis_y.setRange(_min, _max)


class Main(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.db = None
        self.charts = []
        self.backend = Backend()
        self.setMinimumWidth(900)
        self._layout = self.chart_container.layout()
        layout = self.chart_container.layout()

        # Set up the SQLite database connection
        self.setup_database()

        # self.slider.setBackgroundRole()
        self.slider.rootContext().setContextProperty('backend', self.backend)
        self.slider.setSource('ui/slider.qml')
        self.slider.setClearColor(QColor.fromString('transparent'))
        self.addChart('btcusdt', '1min')
        self.addChart('ethusdt', '1min')


        self.backend.updateFirst.connect(self.updateStart)
        self.backend.updateSecond.connect(self.updateEnd)


    def setup_database(self):
        # Set up the SQLite database connection
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("financial_data.db")
        if not self.db.open():
            print("Failed to open database.")
            sys.exit(1)


    def updateStart(self, _start):
        for chart in self.charts:
            chart.start = _start

    def updateEnd(self, _end):
        for chart in self.charts:
            chart.end = _end



    def closeEvent(self, event):
        # Close the database connection when the window is closed
        self.db.close()
        event.accept()

    def addChart(self, symbol, interval):
        chart = CandleChart(symbol, interval)
        self._layout.addWidget(chart)
        self.charts.append(chart)


app = QApplication()
win = Main()
win.show()
app.exec()