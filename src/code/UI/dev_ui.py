# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_dev.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import time
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
##################################################
    def capture_button_clicked(self):
        print("I got a click")
        char = chr(self.char_to_capture.value() + 55)
        frames = self.num_frames.value()
        for i in range(0, frames):
            dc.gatherCameraImage()
            dc.outputClassification(char)
            print("Frame ", str(i + 1), " captured and saved to ", char)
            #time.sleep(0.5)
        print("done")
##################################################

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(475, 115)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 40, 471, 31))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.num_frames = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.num_frames.setMinimum(1)
        self.num_frames.setObjectName("num_frames")
        self.horizontalLayout.addWidget(self.num_frames)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.char_to_capture = QtWidgets.QSpinBox(self.horizontalLayoutWidget)
        self.char_to_capture.setMinimum(10)
        self.char_to_capture.setMaximum(35)
        self.char_to_capture.setProperty("value", 10)
        self.char_to_capture.setDisplayIntegerBase(36)
        self.char_to_capture.setObjectName("char_to_capture")
        self.horizontalLayout.addWidget(self.char_to_capture)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.capture_button = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.capture_button.setObjectName("capture_button")
        self.horizontalLayout.addWidget(self.capture_button)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 10, 401, 31))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.num_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.num_label.setObjectName("num_label")
        self.horizontalLayout_2.addWidget(self.num_label)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem5)
        self.char_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.char_label.setObjectName("char_label")
        self.horizontalLayout_2.addWidget(self.char_label)
        spacerItem6 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem6)
        self.button_label = QtWidgets.QLabel(self.horizontalLayoutWidget_2)
        self.button_label.setObjectName("button_label")
        self.horizontalLayout_2.addWidget(self.button_label)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem7)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 473, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        ####################################################
        self.capture_button.clicked.connect(self.capture_button_clicked)
        ####################################################

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CS 33 Gesture Recognition"))
        self.capture_button.setText(_translate("MainWindow", "Capture"))
        self.num_label.setText(_translate("MainWindow", "Number of Images"))
        self.char_label.setText(_translate("MainWindow", "Character"))
        self.button_label.setText(_translate("MainWindow", "Start"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
