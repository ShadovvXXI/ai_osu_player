from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow
import time as tm
import pygetwindow as gw
import cv2

from song import Song
from utils import window_pos_to_train_pos, draw_image_with_circle

class Recorder(QMainWindow):
    def __init__(self, song_names, camera, img_size, scale, offset):
        super().__init__()
        self.widget = QtWidgets.QLabel(self)
        self.setWindowTitle("My App")
        self.songs = {}
        for name in song_names:
            self.songs[name] = {"file": self.load_song(name)}
        self.camera = camera
        self.img_size = img_size
        self.scale = scale
        self.offset = offset
        self.timer()
        self.start_timer = None
        self.starting = False
        self.recorded_images = dict()
        self.recorded_song_name = ""
        self.training_state = False
        self.training_time = 0
        self.skip_time = 280

    def timer(self):
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.millisecond_tick)
        timer.start(1)

    def starting_skip(self):
        self.start_timer = tm.perf_counter_ns()
        self.training_state = True
        self.recorded_song_name = [s for s in gw.getAllTitles() if "osu!" in s][0]
        self.starting = False

    def millisecond_tick(self):
        # x, y = mouse.get_position()
        # print(f"Курсор находится в точке: ({x}, {y})")
        if "osu!" not in gw.getAllTitles() and any("osu!" in s for s in gw.getAllTitles()):
            if not self.training_state and not self.starting:
                self.starting = True
                timer = QtCore.QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self.starting_skip)
                timer.start(self.skip_time)

        image = self.update_image()
        if self.training_state:
            elapsed_ms = (tm.perf_counter_ns() - self.start_timer) // 1_000_000
            self.recorded_images[elapsed_ms] = image

        if "osu!" in gw.getAllTitles() and self.training_state:
            self.sync_image_to_pos()
            self.recorded_images = dict()
            self.training_state = False
            self.training_time += 1

    # TODO : пересмотреть цветовую гамму (возможно сделать черную)
    def update_image(self):
        res_img = cv2.resize(self.camera.get_latest_frame(), self.img_size, interpolation=cv2.INTER_AREA)

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
            # TODO : при текущей архитектуре новая синхронизация не запускается, возможно проблема с многопоточностью
            new_song.sync_timings_to_pos(self.camera.region, self.scale, self.offset, save_to_file=True)
        return new_song

    def sync_image_to_pos(self):
        for song in self.songs:
            if song.lower() in self.recorded_song_name.lower():
                pos = self.songs[song]["file"].hit_timings_to_pos
                max_pos = max(pos)
                size = (self.camera.region[2] - self.camera.region[0], self.camera.region[3] - self.camera.region[1])
                for moment in sorted(self.recorded_images):
                    timing = (moment - self.songs[song]["file"].lead_in +
                              (self.skip_time*1.3 if self.songs[song]["file"].lead_in else 0))
                    if timing > max_pos:
                        break

                    if timing in pos:
                        current_pos = pos[timing]
                    elif pos[timing-1]:
                        current_pos = pos[timing-1]
                    else:
                        continue

                    self.songs[song][timing] = {
                        "pos": window_pos_to_train_pos(size, current_pos, self.img_size[0], self.img_size[1]),
                        "image": self.recorded_images[moment]
                    }

                    # debug func
                    if moment > 3000:
                        draw_image_with_circle(self.songs[song][timing]["image"], self.songs[song][timing]["pos"])
                break