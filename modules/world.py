import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np
import datetime


# 世界
class World:
    def __init__(self, time_span, time_interval, drawing_range=[-5, 5]):
        self.objects = []
        self.time_span = time_span
        self.time_interval = time_interval
        self.drawing_range = drawing_range

    # 世界にオブジェクトを追加する
    def append(self,obj):
        self.objects.append(obj)

    # 世界を描画する
    def draw(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(self.drawing_range[0], self.drawing_range[1])
        ax.set_ylim(self.drawing_range[0], self.drawing_range[1])
        ax.set_xlabel("X",fontsize=10)
        ax.set_ylabel("Y",fontsize=10)

        elems = []

        self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax),
                                    frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
        plt.show()

    # 世界を 1 ステップ進める
    def one_step(self, i, elems, ax):
        while elems: elems.pop().remove()
        time_str = "t = %.2f[s]" % (self.time_interval*i)
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)
            if hasattr(obj, "one_step"): obj.one_step(self.time_interval)
