import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton
from PySide6.QtCharts import QCandlestickSeries, QCandlestickSet, QCandlestickModelMapper, QChart, QBarCategoryAxis, QValueAxis, QChartView, QDateTimeAxis
from PySide6.QtCore import QDateTime, Qt, QObject, Slot, Property, Signal
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from PySide6.QtQuickWidgets import QQuickWidget
from ui.ui_mainwindow import Ui_Form
from datetime import datetime
import sqlalchemy
from sqlalchemy.sql import text

import db

engine = sqlalchemy.create_engine("sqlite:///./financial_data.db",
                                       execution_options={"sqlite_raw_colnames": True})

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


class Main(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.db = None
        self.charts = []
        self.backend = Backend()
        self.setMinimumWidth(900)
        layout = self.chart_container.layout()

        # Set up the SQLite database connection
        self.setup_database()

        self.slider = QQuickWidget()
        self.slider.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
        self.slider.rootContext().setContextProperty('backend', self.backend)
        self.slider.setSource('ui/slider.qml')
        layout.addWidget(self.slider)
        self.addChart('btcusdt', '1min')
        self.addChart('ethusdt', '1min')


        self.backend.updateFirst.connect(self.updateStart)
        self.backend.updateSecond.connect(self.updateEnd)

        # self.chart.mapToPosition(QPoint(self.series.attachedAxes()))

    def addChart(self, symbol, interval):

        # Create a candlestick chart
        series = QCandlestickSeries()
        chart = QChart()
        chart.setTitle(f"{symbol}_{interval}")
        chart.legend().hide()
        # Create a chart view and add it to the layout
        chart_view = QChartView(chart)
        date_axis = QDateTimeAxis()
        self.charts.append((chart, date_axis))
        self.layout().addWidget(chart_view)
        # Create a candlestick series
        series.setIncreasingColor(Qt.GlobalColor.green)
        series.setDecreasingColor(Qt.GlobalColor.red)
        series.setBodyOutlineVisible(False)

        # Fetch data from the SQLite database
        query = QSqlQuery(f"SELECT Date, Open, High, Low, Close FROM {symbol}_{interval} ORDER BY Date")
        while query.next():
            date = datetime.fromisoformat(query.value("Date"))
            open_price = query.value("Open")
            high_price = query.value("High")
            low_price = query.value("Low")
            close_price = query.value("Close")

            # Create a candlestick set and add it to the series
            candlestick_set = QCandlestickSet(
                open_price, high_price, low_price, close_price,  date.timestamp() * 1000
            )
            series.append(candlestick_set)

        # Add the series to the chart
        chart.addSeries(series)
        chart.createDefaultAxes()

        axis_x = chart.axes(Qt.Orientation.Horizontal)[0]
        axis_x.setVisible(False)
        categories = axis_x.categories()
        start_id = 0
        end_id = -1
        start = QDateTime.fromString(categories[start_id], 'dd.MM.yyyy hh:mm')
        end = QDateTime.fromString(categories[end_id], 'dd.MM.yyyy hh:mm')
        date_axis.setRange(start, end)
        axis_x.setRange(categories[start_id], categories[end_id])
        chart.addAxis(date_axis, Qt.AlignmentFlag.AlignBottom)

        # Format the y-axis
        axis_y = chart.axes(Qt.Orientation.Vertical)[0]
        axis_y.setTitleText("Price")


    def setup_database(self):
        # Set up the SQLite database connection
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("financial_data.db")
        if not self.db.open():
            print("Failed to open database.")
            sys.exit(1)

    def updateStart(self, _start):
        for chart, date_axis in self.charts:
            axis_x = chart.axes(Qt.Orientation.Horizontal)[0]
            categories = axis_x.categories()
            start = int(_start * (len(categories) - 1))
            axis_x.setMin(categories[start])
            start_date = QDateTime.fromString(categories[start], 'dd.MM.yyyy hh:mm')
            date_axis.setMin(start_date)
            y_axis = chart.axes(Qt.Orientation.Vertical)[0]
            y_axis.setRange(*self.get_y_bounds(start_date, date_axis.max(), chart.title()))

    def updateEnd(self, _end):
        for chart, date_axis in self.charts:
            axis_x = chart.axes(Qt.Orientation.Horizontal)[0]
            categories = axis_x.categories()
            end = int(_end * (len(categories) - 1))
            axis_x.setMax(categories[end])
            end_date = QDateTime.fromString(categories[end], 'dd.MM.yyyy hh:mm')
            date_axis.setMax(end_date)
            y_axis = chart.axes(Qt.Orientation.Vertical)[0]
            y_axis.setRange(*self.get_y_bounds(date_axis.min(), end_date, chart.title()))


    def get_y_bounds(self, r0: QDateTime, r1: QDateTime, table_name):
        r0 = r0.toString('yyyy-MM-dd hh:mm:ss')
        r1 = r1.toString('yyyy-MM-dd hh:mm:ss')
        if r0 > r1:
            r0, r1 = (r1, r0)
        with engine.connect() as conn:
            q = text(
                f"SELECT max(High) FROM {table_name} "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _max = conn.execute(q, {'r0': r0, 'r1': r1}).scalar_one()
            q = text(
                f"SELECT min(Low) FROM {table_name} "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _min = conn.execute(q, {'r0': r0, 'r1': r1}).scalar_one()
            # log.info(f'''GET BOUNDS:
            # table   {table_name}
            # range:  {r0} - {r1}
            # bounds: {_min} - {_max}''')
            return _min, _max


    def closeEvent(self, event):
        # Close the database connection when the window is closed
        self.db.close()
        event.accept()


app = QApplication()
win = Main()
win.show()
app.exec()