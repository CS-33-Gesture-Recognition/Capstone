from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication
import sys
import os
import functools


class MLGestureRecognition(object):
    def __init__(self):
        self.setupUI()

    def setupUI(self):
        self.initMainWindow()
        self.initMenuBar()
        self.initUI()
        QtCore.QMetaObject.connectSlotsByName(self.mainWindow)

    def initMainWindow(self):
        self.mainWindow = QtWidgets.QMainWindow()
        self.mainWindow.setGeometry(200, 200, 300, 300)
        self.mainWindow.setWindowTitle("ML Gesture Recognition")
        self.centralWidget = QtWidgets.QWidget(self.mainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.mainWindow.setCentralWidget(self.centralWidget)

    def initMenuBar(self):
        # Create Menu bar
        self.menubar = QtWidgets.QMenuBar(self.mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")

        self.mainWindow.setMenuBar(self.menubar)

        # File dropdown
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuFile.setTitle("File")
        self.menubar.addAction(self.menuFile.menuAction())

        self.actionQuit = QtWidgets.QAction(self.mainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.actionQuit.setText("Quit")
        self.actionQuit.setShortcut("Ctrl+Q")
        self.actionQuit.triggered.connect(lambda: sys.exit())

        self.menuFile.addAction(self.actionQuit)

        # Models dropdown
        self.menuModels = QtWidgets.QMenu(self.menubar)
        self.menuModels.setObjectName("menuModels")
        self.menuModels.setTitle("Models")

        for i in os.listdir("./models"):
            model = QtWidgets.QAction(self.mainWindow)
            model.setObjectName(f"action{i}")
            model.setText(i)
            model.triggered.connect(functools.partial(self.loadModel, i))
            self.menuModels.addAction(model)

        self.menubar.addAction(self.menuModels.menuAction())

        # View dropdown
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuView.setTitle("View")

        for i in ["EndUser", "Dev", "Training"]:
            view = QtWidgets.QAction(self.mainWindow)
            view.setObjectName(f"action{i}")
            view.setText(i)
            view.triggered.connect(functools.partial(self.changeView, i))
            self.menuView.addAction(view)

        self.menubar.addAction(self.menuView.menuAction())

    def initUI(self):
        self.stackedWidget = QtWidgets.QStackedWidget(self.mainWindow)
        self.stackedWidget.setGeometry(QtCore.QRect(200, 200, 300, 300))
        self.stackedWidget.setObjectName("stackedWidget")

        self.widgetEndUser = QtWidgets.QWidget()
        self.widgetEndUser.setObjectName("widgetEndUser")

        self.labelEndUser = QtWidgets.QLabel(self.widgetEndUser)
        self.labelEndUser.setGeometry(150, 150, 100, 10)
        self.labelEndUser.setObjectName("labelEndUser")
        self.labelEndUser.setText("End User")

        self.stackedWidget.addWidget(self.widgetEndUser)

        self.widgetDev = QtWidgets.QWidget()
        self.widgetDev.setObjectName("widgetDev")

        self.labelDev = QtWidgets.QLabel(self.widgetDev)
        self.labelDev.setGeometry(150, 150, 100, 10)
        self.labelDev.setObjectName("labelDev")
        self.labelDev.setText("Dev")

        self.stackedWidget.addWidget(self.widgetDev)

        self.widgetTraining = QtWidgets.QWidget()
        self.widgetTraining.setObjectName("widgetTraining")

        self.labelTraining = QtWidgets.QLabel(self.widgetTraining)
        self.labelTraining.setGeometry(150, 150, 100, 10)
        self.labelTraining.setObjectName("labelTraining")
        self.labelTraining.setText("Training")

        self.stackedWidget.addWidget(self.widgetTraining)

        self.stackedWidget.setCurrentWidget(self.widgetEndUser)

    def debugAction(self, text):
        print(f">>> Debug: {text}")

    def loadModel(self, model):
        print(f">>> Load model: {model}")

    def changeView(self, view):
        print(f">>> Change View: {view}")
        if view == "EndUser":
            self.stackedWidget.setCurrentWidget(self.widgetEndUser)
        elif view == "Dev":
            self.stackedWidget.setCurrentWidget(self.widgetDev)
        elif view == "Training":
            self.stackedWidget.setCurrentWidget(self.widgetTraining)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MLGestureRecognition()
    ui.mainWindow.show()
    sys.exit(app.exec_())
