import sys
from pprint import pformat

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QSizePolicy, QGraphicsScene
from PySide6.QtCharts import QCandlestickSeries, QCandlestickSet, QCandlestickModelMapper, QChart, QValueAxis, QChartView, QDateTimeAxis
from PySide6.QtCore import QDateTime, Qt, QObject, Slot, Property, Signal, QRect
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from PySide6.QtGui import QPalette, QColor, QFont, QIcon

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
        

class CornerWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(150, 100)
        self.setStyleSheet('background-color: black')
        self.raise_()


class CandleChart(QChartView):
    def __init__(self, symbol, interval):
        super().__init__()
        self.table_name = f'{symbol}_{interval}'
        self.symbol = symbol
        self.interval = interval
        self.timedelta = db.DatetimeIntervals[interval]
        self.init_db()

        # Create a candlestick chart
        self._chart = QChart()
        self._chart.setTitle(f"{symbol}_{interval}")
        self._chart.setTitleBrush(QColor.fromString('white'))
        self._chart.legend().hide()
        self._chart.setBackgroundBrush(QColor(50, 50, 50, 255))
        # Create a chart view and add it to the layout
        self.setChart(self._chart)
        size_policy = QSizePolicy()
        size_policy.setVerticalStretch(1)
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
        size_policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.setSizePolicy(size_policy)

        # set scene
        # self.info_widget = CornerWidget(self)
        self.pushButton_close = QPushButton()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.pushButton_close.setSizePolicy(sizePolicy)
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.ApplicationExit))
        self.pushButton_close.setIcon(icon)
        self.pushButton_close.setFlat(True)
        self.pushButton_close.clicked.connect(self._close)
        button = self.scene().addWidget(self.pushButton_close)
        button.setPos(50, 50)




        # Create a candlestick series
        self._series = QCandlestickSeries()
        self._series.setIncreasingColor(Qt.GlobalColor.green)
        self._series.setDecreasingColor(Qt.GlobalColor.red)
        self._series.setBodyOutlineVisible(False)

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
                    self._series.append(candlestick_set)
            prev_date = date
            open_price = query.value("Open")
            high_price = query.value("High")
            low_price = query.value("Low")
            close_price = query.value("Close")

            # Create a candlestick set and add it to the series
            candlestick_set = QCandlestickSet(
                open_price, high_price, low_price, close_price,  date.timestamp() * 1000
            )
            self._series.append(candlestick_set)

        # Add the series to the chart
        self._chart.addSeries(self._series)

        # Create axes
        self._chart.createDefaultAxes()
        self.axis_x = self._chart.axes(Qt.Orientation.Horizontal)[0]
        self.axis_x.setVisible(False)
        categories = self.axis_x.categories()
        start_id = 0
        end_id = -1
        start = QDateTime.fromString(categories[start_id], 'dd.MM.yyyy hh:mm')
        end = QDateTime.fromString(categories[end_id], 'dd.MM.yyyy hh:mm')

        self.date_axis = QDateTimeAxis()
        self.date_axis.setTickCount(8)
        self.date_axis.setRange(start, end)
        self.axis_x.setRange(categories[start_id], categories[end_id])
        self._chart.addAxis(self.date_axis, Qt.AlignmentFlag.AlignBottom)

        # Format the y-axis
        self.axis_y = self._chart.axes(Qt.Orientation.Vertical)[0]
        self.axis_y.setTitleText("Price")
        self.axis_y.setTitleBrush(QColor.fromString('white'))
        self.axis_y.setLabelsColor(QColor.fromString('white'))
        self.date_axis.setLabelsColor(QColor.fromString('white'))

        self.get_y_bounds()

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

    def _close(self):
        self.close()


class Main(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.db = None
        self.charts = []
        self.backend = Backend()
        self.setMinimumWidth(900)
        self._layout = self.chart_container.layout()

        # Set up the SQLite database connection
        self.setup_database()

        # self.slider.setBackgroundRole()
        self.slider.rootContext().setContextProperty('backend', self.backend)
        self.slider.setSource('ui/slider.qml')
        self.slider.setClearColor(QColor.fromString('transparent'))

        self.pushButton_add.clicked.connect(self.addChart)
        self.backend.updateFirst.connect(self.updateStart)
        self.backend.updateSecond.connect(self.updateEnd)
        
    @property
    def symbol(self):
        return self.comboBox_symbol.currentText()
    
    @property
    def interval(self):
        return self.comboBox_interval.currentText()


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

    def addChart(self):
        chart = CandleChart(self.symbol, self.interval)
        self._layout.addWidget(chart)
        self.charts.append(chart)


if __name__ == '__main__':
    app = QApplication()
    win = Main()
    win.show()
    app.exec()