import dxcam
from PyQt6.QtWidgets import QApplication
from win32gui import FindWindow, GetClientRect, ClientToScreen
import sys
import logging

from recorder import Recorder

logging.basicConfig(level=logging.INFO)

# размеры изображения для нейросети
WIDTH = 250
HEIGHT = 125

# обработчик окна и его координаты на экране
window_handle = FindWindow(None, "osu!")
l, t, r, b = GetClientRect(window_handle)
cl, ct = ClientToScreen(window_handle, (l, t))

size = (r - l, b - t)
region = (cl, ct, cl + size[0], ct + size[1])

# по наблюдениям высота поля всегда 80% от общей высоты окна
playfield_h = 0.8 * size[1]
# соотношение поля всегда 3/4
playfield_w = playfield_h * 4/3
# вычисляем scale по длине с наименьшим коэфициентом изменяемости
scale = playfield_h / 384

# расстояние слева и справа всегда одинаковое
offset_x = (size[0] - playfield_w) / 2
# по наблюдениям смещение свеху всегда 11,6% от общей высоты, снизу - 8,3%
offset_y = size[1] * 0.116

# создаем область для записи и начинаем ее
camera = dxcam.create(region=region)
camera.start()

app = QApplication(sys.argv)

# TODO : подумать о возможности переноса всех данных и камеры в рекордер
window = Recorder(["Gira Gira"], camera, img_size=(WIDTH, HEIGHT), scale=scale, offset=(offset_x, offset_y))
window.show()

app.exec()

# окончание записи экрана
camera.stop()