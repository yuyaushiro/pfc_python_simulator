import matplotlib.animation as anm
import datetime


class Recorder:
    def __init__(self, time_interval, file_name, playback_speed=1):
        self.playback_speed = playback_speed

        fps = playback_speed / time_interval

        self.writer = anm.writers['ffmpeg'](fps=fps)
        now = datetime.datetime.now()

        self.file_name = file_name + now.strftime('-%Y%m%d-%H%M%S') + '.mp4'

    def record(self, animation):
        animation.save(self.file_name, writer=self.writer, dpi=250)
