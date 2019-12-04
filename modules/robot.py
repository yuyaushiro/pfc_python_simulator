import matplotlib
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np


# ノイズなしの理想ロボットクラス
class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, size=0.15, color="black"):
        self.pose = np.array(pose)
        self.r = size
        self.color = color
        self.agent = agent
        self.poses = [pose]
        self.sensor = sensor

    # 描画する
    def draw(self, ax, elems):
        x, y, theta = self.pose
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])
        if self.agent and hasattr(self.agent, "draw"):
            self.agent.draw(ax, elems)

    # 状態が遷移する
    @classmethod
    def transition_state(cls, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0),
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )

    # 1 ステップ先に進める
    def one_step(self, time_interval):
        # エージェントが載っていなかったらなにもしない
        if not self.agent: return

        # センサが載っていたら観測する
        observation = None
        if self.sensor:
            observation = self.sensor.observe(self.pose)

        # 行動決定
        nu, omega = self.agent.make_decision(self.pose, observation)

        # ロボットの状態が遷移する
        self.pose = self.transition_state(nu, omega, time_interval, self.pose)

