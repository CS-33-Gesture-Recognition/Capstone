from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit 
import sys
import datacollection as dc

class Container(QWidget):
    
    def __init__(self):
        super().__init__();
        self.initUI();

    def initUI(self):

        contents = "";
        inputText = QLineEdit(contents, self);
        inputText.setToolTip('Input Gesture Classification Here');
        inputText.resize(150, 100);

        captureButton = QPushButton('Capture Data', self);
        captureButton.setToolTip('Use this button to capture an image from camera');
        captureButton.resize(150, 100);
        captureButton.move(150, 0);

    def onCaptureClick():


def main():
    app = QApplication(sys.argv);
    window = Container();
    window.setGeometry(0, 0, 300, 100);
    window.setWindowTitle("Gesture Recognition Training GUI");
    window.show();

    sys.exit(app.exec_());

if __name__ == "__main__":
    main();

