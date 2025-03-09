import sys
from pprint import pformat
from huobi.client.market import MarketClient, Candlestick
from huobi.constant import *

from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QSizePolicy, QVBoxLayout
from PySide6.QtCharts import (QCandlestickSeries, QCandlestickSet, QCandlestickModelMapper, QChart,
                              QChartView, QDateTimeAxis)
from PySide6.QtCore import QDateTime, Qt, Signal, QPoint, QSize
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from PySide6.QtGui import QColor, QFont, QIcon

from ui.ui_mainwindow import Ui_Form
from ui.ui_corner_widget import Ui_cornerWidget
from datetime import datetime, timedelta

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
    CandlestickInterval.MIN1: timedelta(minutes=1),
    CandlestickInterval.MIN5: timedelta(minutes=5),
    CandlestickInterval.MIN15: timedelta(minutes=15),
    CandlestickInterval.MIN30: timedelta(minutes=30),
    CandlestickInterval.MIN60: timedelta(hours=1),
    CandlestickInterval.HOUR4: timedelta(hours=4),
    CandlestickInterval.DAY1: timedelta(days=1),
    CandlestickInterval.WEEK1: timedelta(days=7),
    CandlestickInterval.MON1: timedelta(days=30),
}


market_client = MarketClient(init_log=False)


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

    def setValuse(self, status, _set: QCandlestickSet):
        self.label_open.setText(f'{_set.open()}')
        self.label_low.setText(f'{_set.low()}')
        self.label_high.setText(f'{_set.high()}')
        self.label_close.setText(f'{_set.close()}')
        change = -(_set.open() - _set.close()) / _set.open() * 100
        if change < 0:
            color = 'red'
        else:
            color = 'green'
        self.label_change.setStyleSheet(f'color: {color}')
        self.label_change.setText(f'{change:.2f}%')


class CandleChart(QChartView):
    startChanged = Signal(QDateTime)
    endChanged = Signal(QDateTime)
    rangeChanged = Signal(QDateTime, QDateTime)

    def __init__(self, symbol, interval):
        super().__init__()
        self.table_name = f'{symbol}_{interval}'
        self.symbol = symbol
        self.interval = interval
        self.timeformat = 'dd.MM.yyyy hh:mm'
        self.db_timeformat = 'yyyy-MM-ddThh:mm:ss'
        self.py_db_timeformat = 'Y-M-dd hh:mm:ss'
        self._pressed_pos = QPoint()
        self._min_id = 0
        self._max_id = -1
        self.timedelta = DatetimeIntervals[interval]
        self.item_button = None
        self._series = QCandlestickSeries()
        self._chart = QChart()
        self.axis_x = None
        self.axis_y = None
        self.date_axis = None

        self.init_db()
        self.setup_view()
        self.setup_scene()
        self.setup_chart()
        self.get_y_bounds()

    def setup_view(self):
        self.viewport().setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.ContainsItemShape)
        self.setInteractive(True)
        # self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setOptimizationFlag(self.OptimizationFlag.DontAdjustForAntialiasing)

    def setup_scene(self):
        widget_info = CornerWidget()
        self.scene().addWidget(widget_info)
        margins = self._chart.margins()
        margins.setLeft(margins.left() + widget_info.width())
        self._chart.setMargins(margins)

        pushButton_close = QPushButton()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        pushButton_close.setSizePolicy(sizePolicy)
        icon = QIcon(QIcon.fromTheme(QIcon.ThemeIcon.ApplicationExit))
        pushButton_close.setIcon(icon)
        pushButton_close.setFlat(True)
        pushButton_close.clicked.connect(self._close)
        pushButton_close.setStyleSheet('background-color: transparent')
        self.item_button = self.scene().addWidget(pushButton_close)

        # connect signals
        self._series.hovered.connect(widget_info.setValuse)

    def setup_chart(self):
        self._chart.setTitle(f"{self.symbol}_{self.interval}")
        self._chart.setTitleBrush(QColor.fromString('white'))
        self._chart.legend().hide()
        self._chart.setBackgroundVisible(False)
        self.setChart(self._chart)
        # Create a candlestick series
        self._series.setIncreasingColor(Qt.GlobalColor.green)
        self._series.setDecreasingColor(Qt.GlobalColor.red)
        self._series.setBodyOutlineVisible(False)
        self._series.setMinimumColumnWidth(2)

        # Fetch data from the SQLite database
        query = QSqlQuery(f"SELECT Date, Open, High, Low, Close FROM {self.symbol}_{self.interval} ORDER BY Date")
        prev_date = None
        while query.next():
            if not query.value("Date"):
                continue
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
        start = QDateTime.fromString(categories[start_id], self.timeformat)
        end = QDateTime.fromString(categories[end_id], self.timeformat)

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

        self.startChanged.emit(self.date_axis.min())

    def fetch_data(self, size=2000):
        if size < 1:
            return
        size = min(size, 2000)
        list_obj = market_client.get_candlestick(self.symbol, self.interval, size)
        table_name = f'{self.symbol}_{self.interval}'
        kline: Candlestick
        # log.info('START FETCHING DATA')
        QSqlQuery(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                Date DATETIME PRIMARY KEY,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER
            )
            ''').exec()
        db = QSqlDatabase.database()
        db.transaction()
        for kline in list_obj:
            q = QSqlQuery()
            q.prepare(f'INSERT INTO {table_name} (Date, Open, High, Low, Close, Volume) '
                          f'VALUES (:date, :open, :high, :low, :close, :vol) '
                          f'ON CONFLICT(Date) DO NOTHING;')
            q.bindValue(':date', kline.date.isoformat())
            q.bindValue(':open', kline.open)
            q.bindValue(':high', kline.high)
            q.bindValue(':low', kline.low)
            q.bindValue(':close', kline.close)
            q.bindValue(':vol', kline.vol)
            if not q.exec():
                error = q.lastError().text()
                db.rollback()
                raise Exception(error)
        db.commit()

    def init_db(self):
        query = QSqlQuery(f"SELECT max(Date) as Date FROM {self.table_name}")
        exist = query.next()
        if exist:
            date = query.value('Date')
            last_date = datetime.fromtimestamp(QDateTime.fromString(date, self.db_timeformat).toSecsSinceEpoch())
            now = datetime.now()
            now = round_time(now, self.timedelta)
            intervals_count = 0
            # TODO fix infinite loop (when adding 1 day interval)
            while last_date < now:
                now -= self.timedelta
                intervals_count += 1
        else:
            intervals_count = 2000
        self.fetch_data(intervals_count)

    @property
    def start(self):
        return self.date_axis.min().toString(self.db_timeformat)

    @start.setter
    def start(self, value: QDateTime):
        value = qround_time(value, self.timedelta)
        categories = self.axis_x.categories()
        first = QDateTime.fromString(categories[0], self.timeformat)
        last = QDateTime.fromString(categories[-1], self.timeformat)
        if value < first:
            value = first
        if value > last:
            value = last
        if value >= self.date_axis.max():
            value = QDateTime.fromSecsSinceEpoch((self.date_axis.max().toPython() - self.timedelta).timestamp())
        self.axis_x.setMin(value.toString(self.timeformat))
        self.date_axis.setMin(value)
        self.get_y_bounds()
        self.startChanged.emit(value)

    @property
    def end(self):
        return self.date_axis.max().toString(self.db_timeformat)

    @end.setter
    def end(self, value):
        value = qround_time(value, self.timedelta)
        categories = self.axis_x.categories()
        first = QDateTime.fromString(categories[0], self.timeformat)
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
        self.endChanged.emit(value)

    def get_y_bounds(self):
        q = QSqlQuery()
        q.prepare(
            f"SELECT max(High) AS High FROM {self.table_name} "
            "WHERE Date BETWEEN :r0 AND :r1"
        )
        q.bindValue(':r0', self.start)
        q.bindValue(':r1', self.end)
        q.exec()
        q.first()
        _max = q.value('High')
        q = QSqlQuery()
        q.prepare(
            f"SELECT min(Low) AS Low FROM {self.table_name} "
            "WHERE Date BETWEEN :r0 AND :r1"
        )
        q.bindValue(':r0', self.start)
        q.bindValue(':r1', self.end)
        q.exec()
        q.first()
        _min = q.value('Low')
        if _min and _max:
            self.axis_y.setRange(_min, _max)

    def _close(self):
        self.close()

    def resizeEvent(self, event):
        button_width = self.item_button.size().width()
        button_height = self.item_button.size().height()
        self.item_button.setPos(event.size().width() - button_width - 10, button_height / 2)
        return super().resizeEvent(event)

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
            start_date = QDateTime.fromString(start_category, self.timeformat)
            end_date = QDateTime.fromString(end_category, self.timeformat)
            self.start = start_date
            self.end = end_date
            _range = max(max_offset - min_offset, 1)
            self._series.setBodyWidth(width / _range)
            self.get_y_bounds()
            self.rangeChanged.emit(start_date, end_date)
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
            start_date = QDateTime.fromString(start_category, self.timeformat)
            end_date = QDateTime.fromString(end_category, self.timeformat)
            self.start = start_date
            self.end = end_date
            _range = max(max_offset - min_offset, 1)
            self._series.setBodyWidth(width / _range)
            self.get_y_bounds()
            self.rangeChanged.emit(start_date, end_date)
            return super().mouseMoveEvent(event)
        return super().mouseMoveEvent(event)


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

    def updateRange(self, _start, _end):
        for chart in self.charts:
            if self.sender() == chart:
                continue
            chart.blockSignals(True)
            chart.start = _start
            chart.end = _end
            chart.blockSignals(False)

    def closeEvent(self, event):
        # Close the database connection when the window is closed
        self.db.close()
        event.accept()

    def addChart(self):
        chart = CandleChart(self.symbol, self.interval)
        self._layout.addWidget(chart, stretch=1)
        self.charts.append(chart)
        chart.startChanged.connect(self.dateTimeEdit_start.setDateTime)
        chart.endChanged.connect(self.dateTimeEdit_end.setDateTime)
        chart.rangeChanged.connect(self.updateRange)


if __name__ == '__main__':
    app = QApplication()
    win = Main()
    win.show()
    app.exec()
