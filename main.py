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

from osuparser import beatmapparser, slidercalc
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

def approach_time_ms(ar):
    if ar <= 5:
        return int(1800 - 120 * ar)
    else:
        return int(1200 - 150 * (ar - 5))

class Song:
    def __init__(self, name):
        if name not in os.listdir(osu_songs_directory):
            raise DirectoryWithSongNotFoundError("No such songs")
        self.song_name = name
        self.file = None
        self.parser = None
        self.approach_time = None
        self.part_of_approach_time = None
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
        # время появления объекта до момента его нажатия
        self.approach_time = approach_time_ms(float(self.parser.beatmap["ApproachRate"]))
        self.part_of_approach_time = int(self.approach_time / 10)
        logging.info("Syncing timings to pos started")
        # заполняем время до начала карты центром экрана
        for ms in range(self.parser.beatmap["hitObjects"][0]["startTime"]-self.approach_time):
            self.hit_timings_to_pos[ms] = ((region[2]-region[0])/2, (region[3]-region[1])/2)

        for obj in self.parser.beatmap["hitObjects"]:
            prev_point = self.hit_timings_to_pos[max(self.hit_timings_to_pos.keys())]
            # TODO : сделать правильную обработку карт с появлением первого объекта до начала таймера
            # А кончается Б начинается -> курсор плавно перемещается от А к Б
            # А кончается Б не начинается -> курсор остается в Б
            for moment in range(max(self.hit_timings_to_pos.keys())+1, obj["startTime"]):
                if moment == 10685:
                    print()
                if obj["startTime"] - self.approach_time < moment < obj["startTime"] - self.part_of_approach_time:
                    time_progress = (moment - (obj["startTime"] - self.approach_time)) / self.approach_time

                    cords = osu_cords_to_window_pos(size, obj["position"])
                    cords_progress = (cords[0] - prev_point[0], cords[1] - prev_point[1])

                    x = prev_point[0] + cords_progress[0] * time_progress
                    y = prev_point[0] + cords_progress[1] * time_progress

                    point = (x, y)
                elif moment > obj["startTime"] - self.part_of_approach_time:
                    point = osu_cords_to_window_pos(size, obj["position"])
                else:
                    point = self.hit_timings_to_pos[max(self.hit_timings_to_pos.keys())]
                self.hit_timings_to_pos[moment] = (int(point[0]), int(point[1]))

            # заполняем тайминги объектов
            match obj["object_name"]:
                case 'circle':
                    self.hit_timings_to_pos[obj["startTime"]] = osu_cords_to_window_pos(size, obj["position"])
                case 'slider':
                    for ms in range(obj["duration"] + 1):
                        moment = obj["startTime"] + ms

                        point = slidercalc.get_end_point(obj["curveType"],
                                                         (obj["pixelLength"] * ms / obj["duration"]), obj["points"])
                        if point is None:
                            point = obj["points"][0]

                        self.hit_timings_to_pos[moment] = (osu_cords_to_window_pos(size, point))
                # TODO : сделать обработку для спиннера
        logging.info("Syncing timings to pos ended")

first_song = Song("Rory")
first_song.parse_map_file("Rory")
first_song.build_beatmap()
first_song.sync_timings_to_pos()

app = QApplication(sys.argv)

window = MainWindow()
window.show()
    
app.exec()

# окончание записи экрана
camera.stop()