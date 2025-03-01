import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton
from PySide6.QtCharts import QCandlestickSeries, QCandlestickSet, QCandlestickModelMapper, QChart, QBarCategoryAxis, QValueAxis, QChartView
from PySide6.QtCore import QDateTime, Qt
from PySide6.QtSql import QSqlDatabase, QSqlQueryModel, QSqlQuery
from ui.ui_mainwindow import Ui_Form
from datetime import datetime
import re

class Main(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.db = None
        self.series = None
        layout = self.chart_container.layout()

        # Set up the SQLite database connection
        self.setup_database()

        # Create a candlestick chart
        self.chart = QChart()
        self.chart.setTitle("Candlestick Chart")
        self.chart.legend().hide()

        # Create a chart view and add it to the layout
        self.chart_view = QChartView(self.chart)
        layout.addWidget(self.chart_view)

        # Fetch data from the SQLite database and plot it
        self.fetch_data_and_plot()


    def setup_database(self):
        # Set up the SQLite database connection
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName("financial_data.db")
        if not self.db.open():
            print("Failed to open database.")
            sys.exit(1)

    def fetch_data_and_plot(self):
        # Create a candlestick series
        self.series = QCandlestickSeries()
        self.series.setIncreasingColor(Qt.green)
        self.series.setDecreasingColor(Qt.red)

        # Fetch data from the SQLite database
        query = QSqlQuery("SELECT Date, Open, High, Low, Close FROM btcusdt_1min ORDER BY Date")
        while query.next():
            date = datetime.fromisoformat(query.value("Date"))
            print(date)
            print(date.timestamp())
            open_price = query.value("Open")
            high_price = query.value("High")
            low_price = query.value("Low")
            close_price = query.value("Close")

            # Create a candlestick set and add it to the series
            candlestick_set = QCandlestickSet(
                open_price, high_price, low_price, close_price, date.timestamp()
            )
            self.series.append(candlestick_set)

        # Add the series to the chart
        self.chart.addSeries(self.series)
        self.chart.createDefaultAxes()

        # Format the x-axis to show dates
        axis_x = self.chart.axes(Qt.Orientation.Horizontal)[0]
        # axis_x.setFormat("dd MMM yyyy")
        axis_x.setTitleText("Date")

        # Format the y-axis
        axis_y = self.chart.axes(Qt.Orientation.Vertical)[0]
        axis_y.setTitleText("Price")

    def closeEvent(self, event):
        # Close the database connection when the window is closed
        self.db.close()
        event.accept()


app = QApplication()
win = Main()
win.show()
app.exec()