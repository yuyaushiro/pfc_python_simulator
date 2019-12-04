from modules.world import World, Landmark, Map, Goal
from modules.robot import IdealRobot

import numpy as np


if __name__ == "__main__":
    time_interval = 0.1
    world = World(100, time_interval, drawing_range=[-5, 5])

    ### ランドマーク ###
    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    ### ゴールの追加 ###
    goal = Goal(0.0, 0.0)
    world.append(goal)

    ### ロボットを作る ###
    # 初期位置
    init_pose = np.array([-1.5, -0.5, 0])
    # ロボット
    robot = IdealRobot(init_pose, sensor=None, agent=None)
    world.append(robot)

    world.draw()
