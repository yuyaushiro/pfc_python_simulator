from modules.world import World, Landmark, Map, Goal
from modules.robot import IdealRobot
from modules.agent import Agent, EstimationAgent
from modules.sensor import Camera

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
    # エージェント
    agent = Agent(0.1, 0.5)
    # センサ
    camera = Camera(m)
    # ロボット
    robot = IdealRobot(init_pose, sensor=camera, agent=agent)
    world.append(robot)

    world.draw()
