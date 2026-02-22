from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QMainWindow
from win32gui import FindWindow, GetClientRect, ClientToScreen
import dxcam
import time as tm
import pygetwindow as gw
import cv2
import pickle
import logging

import torch
from torch.utils.data import DataLoader

from song import Song
from nn import OsuImageDataset, OsuNeuralNetwork
from utils import window_pos_to_train_pos, draw_image_with_circle

class Recorder(QMainWindow):
    def __init__(self, song_names, img_size):
        super().__init__()

        # обработчик окна и его координаты на экране
        window_handle = FindWindow(None, "osu!")
        l, t, r, b = GetClientRect(window_handle)
        cl, ct = ClientToScreen(window_handle, (l, t))

        size = (r - l, b - t)
        region = (cl, ct, cl + size[0], ct + size[1])

        # по наблюдениям высота поля всегда 80% от общей высоты окна
        playfield_h = 0.8 * size[1]
        # соотношение поля всегда 3/4
        playfield_w = playfield_h * 4 / 3
        # вычисляем scale по длине с наименьшим коэфициентом изменяемости
        self.scale = playfield_h / 384

        # расстояние слева и справа всегда одинаковое
        offset_x = (size[0] - playfield_w) / 2
        # по наблюдениям смещение свеху всегда 11,6% от общей высоты, снизу - 8,3%
        offset_y = size[1] * 0.116

        self.offset = (offset_x, offset_y)

        # создаем область для записи и начинаем ее
        self.camera = dxcam.create(region=region, output_color="GRAY")
        self.camera.start()

        self.model = OsuNeuralNetwork()
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model.to(device)
        # TODO : функция потерь для 2 измерений
        lr = 1e-3
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # dataset = OsuImageDataset(data)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        self.widget = QtWidgets.QLabel(self)
        self.setWindowTitle("My App")
        self.img_size = img_size
        self.songs = {}
        for name in song_names:
            self.songs[name] = {"file": self.load_song(name)}
        self.timer()
        self.start_timer = None
        self.starting = False
        self.recorded_images = dict()
        self.recorded_song_name = ""
        self.training_state = False
        self.training_time = 0
        self.skip_time = 280

    def __del__(self):
        # окончание записи экрана
        self.camera.stop()

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
            self.sync_image_to_pos(save_to_file=True)
            self.recorded_images = dict()
            self.training_state = False
            self.training_time += 1

    def update_image(self):
        res_img = cv2.resize(self.camera.get_latest_frame(), self.img_size, interpolation=cv2.INTER_AREA)

        # преобразуем в формат подходящий для Qt
        h, w = res_img.shape
        bytes_per_line = w

        qimg = QImage(
            res_img.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_Grayscale8
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
            new_song.sync_timings_to_pos(self.camera.region, self.scale, self.offset, save_to_file=True)
        return new_song

    def sync_image_to_pos(self, save_to_file):
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

                if save_to_file: self.save_to_file(song)
                break

    def save_to_file(self, song_name):
        if len(self.songs[song_name])>2:
            with open("records\\"+song_name+".pkl", "wb") as f:
                pickle.dump(self.songs[song_name], f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_file(self, song_name):
        try:
            with open("records\\"+song_name+".pkl", "rb") as f:
                self.songs[song_name] = pickle.load(f)
                logging.info("Time_to_img_and_pos file loaded")
                return True
        except Exception as e:
            logging.info("Time_to_img_and_pos file corrupted or not find: " + str(e))
            return False