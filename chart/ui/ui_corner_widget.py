# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'corner_widgetjLnlmz.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
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
from PySide6.QtWidgets import (QApplication, QFormLayout, QLabel, QLayout,
    QSizePolicy, QWidget)

class Ui_cornerWidget(object):
    def setupUi(self, cornerWidget):
        if not cornerWidget.objectName():
            cornerWidget.setObjectName(u"cornerWidget")
        cornerWidget.resize(94, 150)
        cornerWidget.setStyleSheet(u"background: transparent")
        self.formLayout = QFormLayout(cornerWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.formLayout.setContentsMargins(0, -1, 0, 0)
        self.label_5 = QLabel(cornerWidget)
        self.label_5.setObjectName(u"label_5")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.label_5)

        self.label = QLabel(cornerWidget)
        self.label.setObjectName(u"label")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.label)

        self.label_2 = QLabel(cornerWidget)
        self.label_2.setObjectName(u"label_2")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.label_2)

        self.label_3 = QLabel(cornerWidget)
        self.label_3.setObjectName(u"label_3")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.label_3)

        self.label_4 = QLabel(cornerWidget)
        self.label_4.setObjectName(u"label_4")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.label_4)

        self.label_6 = QLabel(cornerWidget)
        self.label_6.setObjectName(u"label_6")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.label_6)

        self.label_open = QLabel(cornerWidget)
        self.label_open.setObjectName(u"label_open")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.label_open)

        self.label_high = QLabel(cornerWidget)
        self.label_high.setObjectName(u"label_high")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.label_high)

        self.label_low = QLabel(cornerWidget)
        self.label_low.setObjectName(u"label_low")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.label_low)

        self.label_close = QLabel(cornerWidget)
        self.label_close.setObjectName(u"label_close")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.label_close)

        self.label_change = QLabel(cornerWidget)
        self.label_change.setObjectName(u"label_change")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.label_change)

        self.label_vol = QLabel(cornerWidget)
        self.label_vol.setObjectName(u"label_vol")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.label_vol)


        self.retranslateUi(cornerWidget)

        QMetaObject.connectSlotsByName(cornerWidget)
    # setupUi

    def retranslateUi(self, cornerWidget):
        cornerWidget.setWindowTitle(QCoreApplication.translate("cornerWidget", u"Form", None))
        self.label_5.setText(QCoreApplication.translate("cornerWidget", u"High:", None))
        self.label.setText(QCoreApplication.translate("cornerWidget", u"Low:", None))
        self.label_2.setText(QCoreApplication.translate("cornerWidget", u"Change:", None))
        self.label_3.setText(QCoreApplication.translate("cornerWidget", u"Vol:", None))
        self.label_4.setText(QCoreApplication.translate("cornerWidget", u"Open:", None))
        self.label_6.setText(QCoreApplication.translate("cornerWidget", u"Close:", None))
        self.label_open.setText("")
        self.label_high.setText("")
        self.label_low.setText("")
        self.label_close.setText("")
        self.label_change.setText("")
        self.label_vol.setText("")
    # retranslateUi

