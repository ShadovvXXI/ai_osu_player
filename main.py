from PyQt6.QtWidgets import QApplication
import sys
import logging

from recorder import Recorder

logging.basicConfig(level=logging.INFO)

# размеры изображения для нейросети
WIDTH = 250
HEIGHT = 125

app = QApplication(sys.argv)

window = Recorder(["Gira Gira"], model_name="0.8", img_size=(WIDTH, HEIGHT))
window.show()

app.exec()