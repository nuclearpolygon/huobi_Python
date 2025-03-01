# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindowlMqAsG.ui'
##
## Created by: Qt User Interface Compiler version 6.4.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QHBoxLayout, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget)

class Ui_Form(object):
    def setupUi(self, Form):
        if not Form.objectName():
            Form.setObjectName(u"Form")
        Form.resize(400, 300)
        self.verticalLayout = QVBoxLayout(Form)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.comboBox_symbol = QComboBox(Form)
        self.comboBox_symbol.addItem("")
        self.comboBox_symbol.addItem("")
        self.comboBox_symbol.addItem("")
        self.comboBox_symbol.setObjectName(u"comboBox_symbol")
        sizePolicy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_symbol.sizePolicy().hasHeightForWidth())
        self.comboBox_symbol.setSizePolicy(sizePolicy)

        self.horizontalLayout.addWidget(self.comboBox_symbol)

        self.comboBox_interval = QComboBox(Form)
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.addItem("")
        self.comboBox_interval.setObjectName(u"comboBox_interval")

        self.horizontalLayout.addWidget(self.comboBox_interval)

        self.pushButton_add = QPushButton(Form)
        self.pushButton_add.setObjectName(u"pushButton_add")

        self.horizontalLayout.addWidget(self.pushButton_add)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.chart_container = QWidget(Form)
        self.chart_container.setObjectName(u"chart_container")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.chart_container.sizePolicy().hasHeightForWidth())
        self.chart_container.setSizePolicy(sizePolicy1)
        self.verticalLayout_3 = QVBoxLayout(self.chart_container)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")

        self.verticalLayout.addWidget(self.chart_container)


        self.retranslateUi(Form)

        QMetaObject.connectSlotsByName(Form)
    # setupUi

    def retranslateUi(self, Form):
        Form.setWindowTitle(QCoreApplication.translate("Form", u"Market View", None))
        self.comboBox_symbol.setItemText(0, QCoreApplication.translate("Form", u"btcusdt", None))
        self.comboBox_symbol.setItemText(1, QCoreApplication.translate("Form", u"ethusdt", None))
        self.comboBox_symbol.setItemText(2, QCoreApplication.translate("Form", u"xrpusdt", None))

        self.comboBox_interval.setItemText(0, QCoreApplication.translate("Form", u"1min", None))
        self.comboBox_interval.setItemText(1, QCoreApplication.translate("Form", u"5min", None))
        self.comboBox_interval.setItemText(2, QCoreApplication.translate("Form", u"15min", None))
        self.comboBox_interval.setItemText(3, QCoreApplication.translate("Form", u"30min", None))
        self.comboBox_interval.setItemText(4, QCoreApplication.translate("Form", u"60min", None))
        self.comboBox_interval.setItemText(5, QCoreApplication.translate("Form", u"4hour", None))
        self.comboBox_interval.setItemText(6, QCoreApplication.translate("Form", u"1day", None))
        self.comboBox_interval.setItemText(7, QCoreApplication.translate("Form", u"1week", None))
        self.comboBox_interval.setItemText(8, QCoreApplication.translate("Form", u"1mon", None))

        self.pushButton_add.setText(QCoreApplication.translate("Form", u"Add", None))
    # retranslateUi

