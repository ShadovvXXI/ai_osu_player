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
import pickle

from osuparser import beatmapparser, slidercalc
import os

# import tensorflow as tf
# from tensorflow import keras

logging.basicConfig(level=logging.INFO)

# путь до папки с песнями
osu_songs_directory = os.path.join(os.getenv('LOCALAPPDATA'), 'osu!', 'Songs')

# размеры изображения для нейросети
WIDTH = 250
HEIGHT = 125


def draw_image_with_circle(image, center):
    # радиус круга в пикселях
    radius = 10

    # цвет в формате BGR (чёрный)
    color = (0, 0, 0)

    # толщина -1 означает "залитый круг"
    thickness = -1

    cv2.circle(image, center, radius, color, thickness)

    success = cv2.imwrite("result.jpg", image)
    # TODO : неправильно прописывает координаты, они в осу находятся на другой позиции, UI вне координат
    #  возможно сдвиг тайминга неправильный
    if not success:
        raise IOError("Не удалось сохранить изображение")

class Recorder(QMainWindow):
    def __init__(self, song_names):
        super().__init__()
        self.widget = QtWidgets.QLabel(self)
        self.setWindowTitle("My App")
        self.songs = {}
        for name in song_names:
            self.songs[name] = {"file": self.load_song(name)}
        self.timer()
        self.start_timer = None
        self.recorded_images = dict()
        self.recorded_song_name = None
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
                self.start_timer = tm.perf_counter_ns()
                self.training_state = True
                self.recorded_song_name = [s for s in gw.getAllTitles() if "osu!" in s][0]
            elapsed_ms = (tm.perf_counter_ns() - self.start_timer) // 1_000_000
            self.recorded_images[elapsed_ms] = image

        if "osu!" in gw.getAllTitles() and self.training_state:
            self.sync_image_to_pos()
            self.recorded_images = dict()
            self.training_state = False
            self.training_time += 1

    def update_image(self):
        res_img = cv2.resize(camera.get_latest_frame(), (WIDTH, HEIGHT), cv2.INTER_AREA)
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

    def load_song(self, name):
        new_song = Song(name)
        new_song.parse_map_file(name)
        new_song.build_beatmap()
        if not new_song.load_from_file():
            new_song.sync_timings_to_pos(save_to_file=True)
        return new_song

    def sync_image_to_pos(self):
        for song in self.songs:
            if song in self.recorded_song_name:
                pos = self.songs[song]["file"].hit_timings_to_pos
                max_pos = max(pos)
                for timing in sorted(self.recorded_images):
                    if timing > max_pos:
                        break

                    if timing in pos:
                        current_pos = pos[timing]
                    elif pos[timing-1]:
                        current_pos = pos[timing-1]
                    else:
                        continue

                    self.songs[song][timing] = {
                        "pos": window_pos_to_train_pos(current_pos),
                        "image": self.recorded_images[timing]
                    }

                    if timing > 12000:
                        draw_image_with_circle(self.songs[song][timing]["image"], self.songs[song][timing]["pos"])
                break


def osu_cords_to_window_pos(resolution, cords):
    return int(cords[0] / 512 * resolution[0]), int(cords[1] / 384 * resolution[1])

def window_pos_to_train_pos(pos):
    return int(pos[0] / size[0] * WIDTH), int(pos[1] / size[1] * HEIGHT)


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
    if ar < 5:
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
        self.lead_in = None
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
        # время перед началом песни
        self.lead_in = int(self.parser.beatmap["AudioLeadIn"])
        logging.info("Building done. Time: " + str((tm.perf_counter_ns() - timer_start) // 1_000_000) + "ms")

    # соотносим тайминги с позицией мыши относительно окна
    def sync_timings_to_pos(self, save_to_file):
        # время появления объекта до момента его нажатия
        self.approach_time = approach_time_ms(float(self.parser.beatmap["ApproachRate"]))
        self.part_of_approach_time = int(self.approach_time / 10)
        logging.info("Syncing timings to pos started")
        # заполняем время до начала карты центром экрана
        center = ((region[2] - region[0]) / 2, (region[3] - region[1]) / 2)
        for ms in range(-self.lead_in if self.lead_in else 0,
                        self.parser.beatmap["hitObjects"][0]["startTime"] - self.approach_time):
            self.hit_timings_to_pos[ms] = center

        for obj in self.parser.beatmap["hitObjects"]:
            start_time = obj["startTime"]
            position = obj["position"]
            prev_object_timing = max(self.hit_timings_to_pos)
            prev_point = self.hit_timings_to_pos[prev_object_timing]
            # А кончается Б начинается -> курсор плавно перемещается от А к Б
            # А кончается Б не начинается -> курсор остается в А
            # TODO : ZERO OPTIMIZATION SUPREMACY FUNC - NEED MORE OPTIMIZE IN IF-ELSE - MAYBE SEPARATE IT AND MADE 3 FOR
            for moment in range(prev_object_timing+1, start_time):
                # if moment > 12015:
                #     print()

                if start_time - self.approach_time < moment < start_time - self.part_of_approach_time:
                    time_progress = (moment - (start_time - self.approach_time)) / self.approach_time

                    cords = osu_cords_to_window_pos(size, position)
                    cords_progress = (cords[0] - prev_point[0], cords[1] - prev_point[1])

                    x = prev_point[0] + cords_progress[0] * time_progress
                    y = prev_point[1] + cords_progress[1] * time_progress

                    point = (x, y)
                elif moment > start_time - self.part_of_approach_time:
                    point = osu_cords_to_window_pos(size, position)
                else:
                    point = self.hit_timings_to_pos[prev_object_timing]
                self.hit_timings_to_pos[moment] = (int(point[0]), int(point[1]))

            # заполняем тайминги объекта
            match obj["object_name"]:
                case 'circle':
                    self.hit_timings_to_pos[start_time] = osu_cords_to_window_pos(size, position)
                case 'slider':
                    for ms in range(obj["duration"] + 1):
                        moment = start_time + ms

                        point = slidercalc.get_end_point(obj["curveType"],
                                                         (obj["pixelLength"] * ms / obj["duration"]), obj["points"])
                        if point is None:
                            point = obj["points"][0]

                        self.hit_timings_to_pos[moment] = (osu_cords_to_window_pos(size, point))
                # TODO : сделать обработку для спиннера
        if save_to_file: self.save_to_file()
        logging.info("Syncing timings to pos ended")

    def save_to_file(self):
        if self.hit_timings_to_pos:
            with open(self.song_name+".pkl", "wb") as f:
                pickle.dump(self.hit_timings_to_pos, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_file(self):
        try:
            with open(self.song_name+".pkl", "rb") as f:
                self.hit_timings_to_pos = pickle.load(f)
                return True
        except Exception as e:
            # TODO : выводится позже необходимого, изменить логику
            print("Time_to_pos file corrupted or not find:", e)
            return False


app = QApplication(sys.argv)

window = Recorder(["Rory"])
window.show()
    
app.exec()

# окончание записи экрана
camera.stop()