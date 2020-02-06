# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'end_user_ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

from __future__ import print_function, division

import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import datacollection as dc;
import numpy as np;
import torch
from torch.utils import data
import datacollection as dc
import time
import backgroud_rc
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

# const strings
image_01 = "rgb_image.jpg"
image_02 = "depth_image.jpg"
image_width = 320
image_height = 240
capture_button_text = "Capture"
info_button_text = "Notice"
instruction_text = "This is the user interface. Click ‘capture’ to start shooting. Make sure the user ’s hand is within the range of the camera when shooting. The results will be displayed in the upper right corner."
window_title_text = "CS-33 Gesture Recognition"


class Ui_MainWindow1(object):

    def prosess_image(self):
        print("capturing data")

        test_data = dc.collectTestingX();
        print("done gathering")
        three_channel_data = []
        three_channel_data.append([test_data, test_data, test_data])
        ml_input = torch.Tensor(three_channel_data).to(self.device)

        print("thinking")
        output = self.model(ml_input);

        prediction = int(torch.max(output.data, 1)[1])
        print("done")
        print('updating images')
        pixmap01 = QPixmap(image_01).scaled( image_width, image_height, Qt.KeepAspectRatio)
        pixmap02 = QPixmap(image_02).scaled( image_width, image_height, Qt.KeepAspectRatio)
        self.image01.setPixmap(pixmap01)
        self.image02.setPixmap(pixmap02)
        print('images updated');
        return prediction;

    def capture_button_clicked(self):
        prediction = self.prosess_image()

        pixmap01 = QPixmap(image_01).scaled( image_width, image_height, Qt.KeepAspectRatio)
        pixmap02 = QPixmap(image_02).scaled( image_width, image_height, Qt.KeepAspectRatio)
        self.image01.setPixmap(pixmap01)
        self.image02.setPixmap(pixmap02)
        self.output_text.setText(str(chr(prediction+97)))
        self.ML_output_text.setText("Testing ML")

    def setup(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 480)
        MainWindow.setStyleSheet("background-image: url(:/background/1.jpg);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image01 = QtWidgets.QLabel(self.centralwidget)
        self.image01.setGeometry(QtCore.QRect(15, 15, image_width, image_height))
        self.image01.setText("")
        self.image01.setObjectName("image01")
        self.image02 = QtWidgets.QLabel(self.centralwidget)
        self.image02.setGeometry(QtCore.QRect(350, 15, image_width, image_height))
        self.image02.setText("")
        self.image02.setObjectName("image02")
        self.output_text = QtWidgets.QLabel(self.centralwidget)
        self.output_text.setGeometry(QtCore.QRect(685, 15, 200, 60))
        font = QtGui.QFont()
        font.setPointSize(55)
        self.output_text.setFont(font)
        self.output_text.setText("")
        self.output_text.setAlignment(QtCore.Qt.AlignCenter)
        self.output_text.setObjectName("output_text")
        self.ML_output_text = QtWidgets.QLabel(self.centralwidget)
        self.ML_output_text.setGeometry(QtCore.QRect(685, 90, 200, 60))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ML_output_text.setFont(font)
        self.ML_output_text.setText("")
        self.ML_output_text.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.ML_output_text.setObjectName("ML_output_text")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(690, 250, 171, 51))
        self.pushButton.setStyleSheet("background-color:red;\n"
"border-style:outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font:bold 14px;\n"
"padding:10px;\n"
"min-width:10px;")
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName("menubar")
        self.Notice = QtWidgets.QPushButton(self.centralwidget)
        self.Notice.setGeometry(QtCore.QRect(800, 400, 81, 31))
        self.Notice.setStyleSheet("background-color:red;\n"
"border-style:outset;\n"
"border-width:2px;\n"
"border-radius:10px;\n"
"border-color:black;\n"
"font:bold 10px;\n"
"padding:6px;\n"
"min-width:10px;")
        self.Notice.setObjectName("Notice")
        self.Notice.clicked.connect(self.show_popup)
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #################################################################
        self.pushButton.clicked.connect(self.capture_button_clicked)
        #################################################################

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", window_title_text))
        self.pushButton.setText(_translate("MainWindow", capture_button_text))
        self.Notice.setText(_translate("MainWindow", info_button_text))

    def show_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle(info_button_text)
        msg.setText(instruction_text)
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()

    def importModel(self):
        print('begin loading');
        y = dc.collectTrainingY();
        num_labels = len(set(y));

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        print(self.device);

        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model.fc = nn.Linear(num_ftrs, num_labels)
        self.model = self.model.to(self.device)

        trained_model = torch.load('trained_model.pth.tar', map_location=self.device)
        self.model.load_state_dict(trained_model['state_dict'])

        self.model.eval();
        print('done loading')


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow1()
    ui.setup( MainWindow)
    ui.importModel();
    MainWindow.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
