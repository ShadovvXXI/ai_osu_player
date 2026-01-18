from tkinter.font import names

import dxcam
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow
from win32gui import FindWindow, GetClientRect, ClientToScreen
import pygetwindow as gw
import cv2
import sys
import mouse
import time as tm
import logging

from osuparser import beatmapparser
from random import shuffle
import os

# import tensorflow as tf
# from tensorflow import keras

logging.basicConfig(level=logging.INFO)

# путь до папки с песнями
osu_songs_directory = os.path.join(os.getenv('LOCALAPPDATA'), 'osu!', 'Songs')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.widget = QtWidgets.QLabel(self)
        self.setWindowTitle("My App")
        self.timer()
        self.timer_start = None
        self.recorded_image = dict()
        self.training_state = False
        self.training_time = 0

    def timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.millisecond_tick)
        timer.start(1)

    def millisecond_tick(self):
        # x, y = mouse.get_position()
        # print(f"Курсор находится в точке: ({x}, {y})")

        image = self.update_image()
        if "osu!" not in gw.getAllTitles() and any("osu!" in s for s in gw.getAllTitles()):
            if not self.training_state:
                self.timer_start = tm.perf_counter_ns()
                self.training_state = True
            elapsed_ms = (tm.perf_counter_ns() - self.timer_start) // 1_000_000
            self.recorded_image[elapsed_ms] = image

        if "osu!" in gw.getAllTitles() and self.training_state:
            self.training_state = False
            self.training_time += 1

    def update_image(self):
        res_img = cv2.resize(camera.get_latest_frame(), (250, 125), cv2.INTER_AREA)
        # Image.fromarray(res_img).show()

        # преобразуем в формат подходящий для Qt
        h, w, ch = res_img.shape
        bytes_per_line = ch * w

        qimg = QImage(
            res_img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimg)
        self.widget.setScaledContents(True)
        self.widget.setPixmap(pixmap)

        self.setCentralWidget(self.widget)

        return res_img


def osu_cords_to_window_pos(resolution, cords):
    return int(cords[0] / 512 * resolution[0]), int(cords[1] / 384 * resolution[1])


# обработчик окна и его координаты на экране
window_handle = FindWindow(None, "osu!")
l, t, r, b = GetClientRect(window_handle)
cl, ct = ClientToScreen(window_handle, (l, t))

size = (r - l, b - t)
region = (cl, ct, cl + size[0], ct + size[1])

# создаем область для записи и начинаем ее
camera = dxcam.create()
camera.start(region=region)


class DirectoryWithSongNotFoundError(Exception):
    pass

class MapFileNotFoundError(Exception):
    pass

class Song:
    def __init__(self, name):
        if name not in os.listdir(osu_songs_directory):
            raise DirectoryWithSongNotFoundError("No such songs")
        self.song_name = name
        self.file = None
        self.parser = None
        self.hit_timings_to_pos = dict()

    def parse_map_file(self, map_name):
        song_directory = os.path.join(osu_songs_directory, self.song_name)
        maps = [x for x in os.listdir(song_directory)
                if map_name in x and x.endswith(".osu")]
        if not maps:
            raise MapFileNotFoundError("No such map")

        self.file = maps[0]
        osu_path = os.path.join(song_directory, self.file)
        self.parser = beatmapparser.BeatmapParser()

        timer_start = tm.perf_counter_ns()
        logging.info("Parsing started")
        self.parser.parseFile(osu_path)
        logging.info("Parsing done. Time: " + str((tm.perf_counter_ns() - timer_start) // 1_000_000) + "ms")

    def build_beatmap(self):
        timer_start = tm.perf_counter_ns()
        logging.info("Map building started")
        self.parser.build_beatmap()
        logging.info("Building done. Time: " + str((tm.perf_counter_ns() - timer_start) // 1_000_000) + "ms")

    # соотносим тайминги с позицией мыши относительно окна
    def sync_timings_to_pos(self):
        for obj in self.parser.beatmap["hitObjects"]:
            match obj["object_name"]:
                case 'circle':
                    self.hit_timings_to_pos[obj["startTime"]] = osu_cords_to_window_pos(size, obj["position"])
                case 'slider':
                    slider_len = len(obj["points"])
                    for point_i in range(slider_len-1):
                        interval = obj["duration"] / (slider_len-1)
                        spacing = (obj["points"][point_i][0] - obj["points"][point_i+1][0],
                                   obj["points"][point_i][1] - obj["points"][point_i+1][1])
                        for x in range(int(interval)):
                            cords_in_x = (obj["points"][point_i][0] - spacing[0]*(x/interval),
                                          obj["points"][point_i][1] - spacing[1]*(x/interval))
                            self.hit_timings_to_pos[round(obj["startTime"] + point_i*interval + x)] = (
                                osu_cords_to_window_pos(size, cords_in_x))
                # TODO : сделать обработку для спиннера

first_song = Song("Gira Gira")
first_song.parse_map_file("Gira Gira")
first_song.build_beatmap()
first_song.sync_timings_to_pos()

app = QApplication(sys.argv)

window = MainWindow()
window.show()
    
app.exec()

# окончание записи экрана
camera.stop()