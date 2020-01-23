

import datacollection as dc
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

str_image01 = "image.jpeg"
str_image02 = "image.jpeg"
str_win_title = "CS 33 Gesture Recognition"
str_cap_btn = "Capture"
image_width = 320
image_height = 240
window_width = 900
window_height = 280


class Ui_MainWindow(object):
    def capture_button_clicked(self):
        # define capture sequence
        print("capture")
        self.update_Ui("B", "ML was 100% accurate")

    def update_Ui(self, ML_output, ML_output_extra):
        #define update sequence
        print("update")
        pixmap01 = QPixmap(str_image01).scaled( image_width, image_height, Qt.KeepAspectRatio)
        pixmap02 = QPixmap(str_image02).scaled( image_width, image_height, Qt.KeepAspectRatio)
        self.image01.setPixmap(pixmap01)
        self.image02.setPixmap(pixmap02)
        self.output_text.setText(ML_output)
        self.ML_output_text.setText(ML_output_extra)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(window_width, window_height)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.image01 = QtWidgets.QLabel(self.centralwidget)
        self.image01.setGeometry(QtCore.QRect(15, 15, 320, 240))
        self.image01.setText("")
        #pixmap01 = QPixmap(str_image01).scaled( image_width, image_height, Qt.KeepAspectRatio)
        #self.image01.setPixmap(pixmap01)
        self.image01.setObjectName("image01")
        self.image02 = QtWidgets.QLabel(self.centralwidget)
        self.image02.setGeometry(QtCore.QRect(350, 15, 320, 240))
        self.image02.setText("")
        #pixmap02 = QPixmap(str_image02).scaled( image_width, image_height, Qt.KeepAspectRatio)
        #self.image02.setPixmap(pixmap02)
        self.image02.setObjectName("image02")
        self.output_text = QtWidgets.QLabel(self.centralwidget)
        self.output_text.setGeometry(QtCore.QRect(685, 15, 200, 60))
        font = QtGui.QFont()
        font.setPointSize(55)
        self.output_text.setFont(font)
        self.output_text.setText("")
        self.output_text.setAlignment(QtCore.Qt.AlignCenter)
        self.output_text.setObjectName("output_text")
        self.capture_button = QtWidgets.QPushButton(self.centralwidget)
        self.capture_button.setGeometry(QtCore.QRect(685, 195, 200, 60))
        self.capture_button.setObjectName("capture_button")
        self.ML_output_text = QtWidgets.QLabel(self.centralwidget)
        self.ML_output_text.setGeometry(QtCore.QRect(685, 90, 200, 60))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.ML_output_text.setFont(font)
        self.ML_output_text.setText("")
        self.ML_output_text.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.ML_output_text.setObjectName("ML_output_text")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        #################################################################
        self.capture_button.clicked.connect(self.capture_button_clicked)
        #################################################################
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", str_win_title))
        self.capture_button.setText(_translate("MainWindow", str_cap_btn))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
