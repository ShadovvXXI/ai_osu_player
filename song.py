from osuparser import beatmapparser, slidercalc
import os
import time as tm
import logging
import pickle

from utils import osu_cords_to_window_pos, approach_time_ms

# путь до папки с песнями
osu_songs_directory = os.path.join(os.getenv('LOCALAPPDATA'), 'osu!', 'Songs')

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
    def sync_timings_to_pos(self, region, scale, offset, save_to_file):
        # время появления объекта до момента его нажатия
        self.approach_time = approach_time_ms(float(dict(self.parser.beatmap)["ApproachRate"]))
        self.part_of_approach_time = int(self.approach_time / 10)
        logging.info("Syncing timings to pos started")
        # заполняем время до начала карты центром экрана
        center = ((region[2] - region[0]) / 2, (region[3] - region[1]) / 2)
        for ms in range(-self.lead_in if self.lead_in else 0,
                        self.parser.beatmap["hitObjects"][0]["startTime"] - self.approach_time):
            self.hit_timings_to_pos[ms] = center

        for obj in self.parser.beatmap["hitObjects"]:
            obj_start_time = obj["startTime"]
            position = obj["position"]
            prev_object_timing = max(self.hit_timings_to_pos)
            prev_point = self.hit_timings_to_pos[prev_object_timing]
            approach_start_time = min(obj_start_time-prev_object_timing, self.approach_time)

            # А кончается Б начинается -> курсор плавно перемещается от А к Б
            # А кончается Б не начинается -> курсор остается в А
            for moment in range(prev_object_timing+1, obj_start_time - approach_start_time + 1):
                point = self.hit_timings_to_pos[prev_object_timing]
                self.hit_timings_to_pos[moment] = (int(point[0]), int(point[1]))

            cords = osu_cords_to_window_pos(position, scale, offset)
            cords_progress = (cords[0] - prev_point[0], cords[1] - prev_point[1])
            for moment in range(obj_start_time - approach_start_time + 1, obj_start_time - self.part_of_approach_time + 1):
                # if moment > 12177:
                #     print(sep="", end="")
                time_progress = ((moment - (obj_start_time - approach_start_time))
                                 / (approach_start_time - self.part_of_approach_time))

                x = prev_point[0] + cords_progress[0] * time_progress
                y = prev_point[1] + cords_progress[1] * time_progress

                self.hit_timings_to_pos[moment] = (int(x), int(y))

            for moment in range(obj_start_time - self.part_of_approach_time + 1, obj_start_time):
                point = osu_cords_to_window_pos(position, scale, offset)
                self.hit_timings_to_pos[moment] = (int(point[0]), int(point[1]))

            # заполняем тайминги объекта
            match obj["object_name"]:
                case 'circle':
                    self.hit_timings_to_pos[obj_start_time] = osu_cords_to_window_pos(position, scale, offset)
                case 'slider':
                    for ms in range(obj["duration"] + 1):
                        moment = obj_start_time + ms

                        point = slidercalc.get_end_point(obj["curveType"],
                                                         (obj["pixelLength"] * ms / obj["duration"]), obj["points"])
                        if point is None:
                            point = obj["points"][0]

                        self.hit_timings_to_pos[moment] = osu_cords_to_window_pos(point, scale, offset)
                # TODO : сделать обработку для спиннера
        if save_to_file: self.save_to_file()
        logging.info("Syncing timings to pos ended")

    def save_to_file(self):
        if self.hit_timings_to_pos:
            with open("songs\\"+self.song_name+".pkl", "wb") as f:
                pickle.dump(self.hit_timings_to_pos, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_from_file(self):
        try:
            with open("songs\\"+self.song_name+".pkl", "rb") as f:
                self.hit_timings_to_pos = pickle.load(f)
                logging.info("Time_to_pos file loaded")
                return True
        except Exception as e:
            logging.info("Time_to_pos file corrupted or not find: " + str(e))
            return False