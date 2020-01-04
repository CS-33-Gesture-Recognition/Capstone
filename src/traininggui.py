#PyQt imports
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit 

import sys
import datacollection as dc

class Container(QWidget):

    def __init__(self):
        super().__init__();
        self.__textField = "";
        self.initUI();

    def initUI(self):

        inputText = QLineEdit(self.__textField, self);
        inputText.setToolTip('Input Gesture Classification Here');
        inputText.resize(150, 100);
        inputText.textChanged.connect(self.onTextChange);

        captureButton = QPushButton('Capture Data', self);
        captureButton.setToolTip('Use this button to capture an image from camera');
        captureButton.resize(150, 100);
        captureButton.move(150, 0);
        captureButton.clicked.connect(self.onCaptureClick);

    def onTextChange(self, text):
        self.__textField = text;
        return;


    def onCaptureClick(self):
        dc.gatherCameraImage();
        dc.outputClassification(self.__textField);
        print("self.__textField ", self.__textField);
        return;



def main():
    app = QApplication(sys.argv);
    window = Container();
    window.setGeometry(0, 0, 300, 100);
    window.setWindowTitle("Gesture Recognition Training GUI");
    window.show();

    sys.exit(app.exec_());

if __name__ == "__main__":
    main();

