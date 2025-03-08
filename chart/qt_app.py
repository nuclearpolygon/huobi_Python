import sys
from pprint import pformat

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QSizePolicy, QGraphicsScene
from PySide6.QtCharts import QCandlestickSeries, QCandlestickSet, QCandlestickModelMapper, QChart, QValueAxis, QChartView, QDateTimeAxis
from PySide6.QtCore import QDateTime, Qt, QObject, Slot, Property, Signal, QRect, QMargins, QPoint
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from PySide6.QtGui import QPalette, QColor, QFont, QIcon

from ui.ui_mainwindow import Ui_Form
from ui.ui_corner_widget import Ui_cornerWidget
from datetime import datetime, timedelta
import sqlalchemy
from sqlalchemy.sql import text

import db

engine = sqlalchemy.create_engine("sqlite:///./financial_data.db",
                                       execution_options={"sqlite_raw_colnames": True})


def round_time(now: datetime, _timedelta: timedelta) -> datetime:
    return datetime.fromtimestamp(now.timestamp() - (now.timestamp() % _timedelta.total_seconds()))

def qround_time(time: QDateTime, _timedelta: timedelta) -> QDateTime:
    rounded = round_time(time.toPython(), _timedelta).timestamp()
    return QDateTime.fromSecsSinceEpoch(int(rounded))


def time_range(start: datetime, end: datetime, _timedelta: timedelta):
    while start + _timedelta != end - _timedelta:
        start += _timedelta
        yield start


class CornerWidget(QWidget, Ui_cornerWidget):
    def __init__(self, ):
        super().__init__()
        self.setupUi(self)


class CandleChart(QChartView):
    _startChanged = Signal(QDateTime)
    _endChanged = Signal(QDateTime)

    def __init__(self, symbol, interval):
        super().__init__()
        self.table_name = f'{symbol}_{interval}'
        self.symbol = symbol
        self.interval = interval
        self.timeformat = 'dd.MM.yyyy hh:mm'
        self.bd_timeformat = 'yyyy-MM-dd hh:mm:ss'
        self._pressed_pos = QPoint()
        self._min_id = 0
        self._max_id = -1
        self.timedelta = db.DatetimeIntervals[interval]
        self.init_db()

        self.viewport().setContentsMargins(0,0,0,0)
        self.setContentsMargins(0,0,0,0)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.ContainsItemShape)
        self.setInteractive(True)
        # self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setOptimizationFlag(self.OptimizationFlag.DontAdjustForAntialiasing)

        # Create a candlestick chart
        self._chart = QChart()
        self._chart.setTitle(f"{symbol}_{interval}")
        self._chart.setTitleBrush(QColor.fromString('white'))
        self._chart.legend().hide()
        self._chart.setBackgroundVisible(False)
        # self._chart.setBackgroundBrush(QColor(50, 50, 50, 255))
        # Create a chart view and add it to the layout
        self.setChart(self._chart)
        size_policy = QSizePolicy()
        size_policy.setVerticalStretch(1)
        size_policy.setHorizontalPolicy(QSizePolicy.Policy.Preferred)
        size_policy.setVerticalPolicy(QSizePolicy.Policy.Minimum)
        self.setSizePolicy(size_policy)

        # set scene
        self.widget_info = CornerWidget()
        self.item_info = self.scene().addWidget(self.widget_info)
        margins = self._chart.margins()
        margins.setLeft(margins.left() + self.widget_info.width())
        self._chart.setMargins(margins)

        self.pushButton_close = QPushButton()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.pushButton_close.setSizePolicy(sizePolicy)
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.ApplicationExit))
        self.pushButton_close.setIcon(icon)
        self.pushButton_close.setFlat(True)
        self.pushButton_close.clicked.connect(self._close)
        self.pushButton_close.setStyleSheet('background-color: transparent')
        self.item_button = self.scene().addWidget(self.pushButton_close)




        # Create a candlestick series
        self._series = QCandlestickSeries()
        self._series.setIncreasingColor(Qt.GlobalColor.green)
        self._series.setDecreasingColor(Qt.GlobalColor.red)
        self._series.setBodyOutlineVisible(False)
        self._series.hovered.connect(self.onHovered)

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
        self.date_axis.setLabelsFont(QFont('mono', 10))

        self._startChanged.emit(self.date_axis.min())
        self.get_y_bounds()

    def init_db(self):
        query = QSqlQuery(f"SELECT max(Date) as Date FROM {self.table_name}")
        query.next()
        date = query.value('Date')
        last_date = datetime.fromtimestamp(QDateTime.fromString(date, self.bd_timeformat).toSecsSinceEpoch())
        now = datetime.now()
        now = round_time(now, self.timedelta)
        intervals_count = 0
        while last_date < now:
            now -= self.timedelta
            intervals_count += 1
        db.fetch_data(self.symbol, self.interval, intervals_count)

    @property
    def start(self):
        return self.date_axis.min().toString(self.bd_timeformat)

    @start.setter
    def start(self, value: QDateTime):
        value = qround_time(value, self.timedelta)
        categories = self.axis_x.categories()
        first = QDateTime.fromString(categories[0], self.timeformat)
        last = QDateTime.fromString(categories[-2], self.timeformat)
        if value < first:
            value = first
        if value > last:
            value = last
        if value >= self.date_axis.max():
            value = QDateTime.fromSecsSinceEpoch((self.date_axis.max().toPython() - self.timedelta).timestamp())
        self.axis_x.setMin(value.toString(self.timeformat))
        self.date_axis.setMin(value)
        self.get_y_bounds()
        # self._startChanged.emit(value)

    @property
    def end(self):
        return self.date_axis.max().toString(self.bd_timeformat)

    @end.setter
    def end(self, value):
        value = qround_time(value, self.timedelta)
        categories = self.axis_x.categories()
        first = QDateTime.fromString(categories[1], self.timeformat)
        last = QDateTime.fromString(categories[-1], self.timeformat)
        if value > last:
            value = last
        if value < first:
            value = first
        if value <= self.date_axis.min():
            value = QDateTime.fromSecsSinceEpoch((self.date_axis.max().toPython() + self.timedelta).timestamp())
        self.axis_x.setMax(value.toString(self.timeformat))
        self.date_axis.setMax(value)
        self.get_y_bounds()
        # self._endChanged.emit(value)

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

    def resizeEvent(self, event):
        button_width = self.item_button.size().width()
        button_height = self.item_button.size().height()
        self.item_button.setPos(event.size().width() - button_width - 10, button_height / 2)
        return super().resizeEvent(event)

    def onHovered(self, status, _set: QCandlestickSet):
        self.widget_info.label_open.setText(f'{_set.open()}')
        self.widget_info.label_low.setText(f'{_set.low()}')
        self.widget_info.label_high.setText(f'{_set.high()}')
        self.widget_info.label_close.setText(f'{_set.close()}')

    def mousePressEvent(self, event):
        self._pressed_pos = event.scenePosition()
        self._min_id = self.axis_x.categories().index(self.axis_x.min())
        self._max_id = self.axis_x.categories().index(self.axis_x.max())
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        width = self._chart.plotArea().width()
        event_pos = event.scenePosition().x()
        offset = (event_pos - self._pressed_pos.x()) / width
        offset_id = int((self._max_id - self._min_id) * offset)
        max_id = self.axis_x.count() - 1
        if event.buttons() == Qt.MouseButton.LeftButton:
            min_offset = min(max(self._min_id - offset_id, 0), max_id)
            max_offset = min(max(self._max_id - offset_id, 0), max_id)
            start_category = self.axis_x.at(min_offset)
            end_category = self.axis_x.at(max_offset)
            self.axis_x.setMin(start_category)
            self.axis_x.setMax(end_category)
            start_date = QDateTime.fromString(start_category, self.timeformat)
            end_date = QDateTime.fromString(end_category, self.timeformat)
            self.date_axis.setMin(start_date)
            self.date_axis.setMax(end_date)
            self.get_y_bounds()
            return super().mouseMoveEvent(event)
        if event.buttons() == Qt.MouseButton.RightButton:
            if offset_id > 1:
                min_offset = min(max(self._min_id + offset_id, 0), max_id)
                max_offset = min(max(self._max_id - offset_id, 0), max_id)
            else:
                min_offset = min(max(self._min_id + offset_id, 0), max_id)
                max_offset = min(max(self._max_id - offset_id, 0), max_id)
            start_category = self.axis_x.at(min_offset)
            end_category = self.axis_x.at(max_offset)
            self.axis_x.setMin(start_category)
            self.axis_x.setMax(end_category)
            start_date = QDateTime.fromString(start_category, self.timeformat)
            end_date = QDateTime.fromString(end_category, self.timeformat)
            self.date_axis.setMin(start_date)
            self.date_axis.setMax(end_date)
            self.get_y_bounds()
            return super().mouseMoveEvent(event)
        return super().mouseMoveEvent(event)
        # start_category = self.axis_x.


class Main(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.db = None
        self.charts = []
        self.setMinimumWidth(900)
        self._layout = self.chart_container.layout()

        # Set up the SQLite database connection
        self.setup_database()
        now = int(datetime.now().timestamp())
        self.dateTimeEdit_end.setDateTime(QDateTime.fromSecsSinceEpoch(now))

        self.pushButton_add.clicked.connect(self.addChart)
        self.dateTimeEdit_start.dateTimeChanged.connect(self.updateStart)
        self.dateTimeEdit_end.dateTimeChanged.connect(self.updateEnd)

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
            chart.blockSignals(True)
            chart.start = _start
            chart.blockSignals(False)

    def updateEnd(self, _end):
        for chart in self.charts:
            chart.blockSignals(True)
            chart.end = _end
            chart.blockSignals(False)

    def closeEvent(self, event):
        # Close the database connection when the window is closed
        self.db.close()
        event.accept()

    def addChart(self):
        chart = CandleChart(self.symbol, self.interval)
        self._layout.addWidget(chart)
        self.charts.append(chart)
        chart._startChanged.connect(self.dateTimeEdit_start.setDateTime)
        chart._endChanged.connect(self.dateTimeEdit_end.setDateTime)


if __name__ == '__main__':
    app = QApplication()
    win = Main()
    win.show()
    app.exec()