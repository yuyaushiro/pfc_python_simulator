from modules.world import World, Landmark, Map, Goal
from modules.robot import IdealRobot
from modules.agent import Agent, EstimationAgent
from modules.sensor import Camera
from modules.mcl import Mcl

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
    # 初期位置のばらつき
    init_pose_stds = np.array([0.03, 0.03, 0.03])
    # 動作のばらつき
    motion_noise_stds = {"nn":0.03, "no":0.03, "on":0.03, "oo":0.03}

    # 推定器
    estimator = Mcl(m, init_pose, 100, motion_noise_stds=motion_noise_stds,
                    init_pose_stds=init_pose_stds)
    # エージェント
    agent = EstimationAgent(time_interval, 0.1, 0.5, estimator)
    # ロボット
    robot = IdealRobot(init_pose, sensor=Camera(m), agent=agent)
    world.append(robot)

    world.draw()
