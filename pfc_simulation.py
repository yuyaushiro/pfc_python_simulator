from modules.world import World, Landmark, Map, Goal
from modules.robot import IdealRobot
from modules.agent import Agent, EstimationAgent
from modules.sensor import Camera
from modules.mcl import Mcl
from modules.grid_map import GridMap
from modules.pfc import Pfc
from modules.recorder import Recorder

import numpy as np


if __name__ == "__main__":
    time_interval = 0.1
    world = World(120, time_interval,
                #   Recorder(time_interval, "avoid1000", playback_speed=3),
                  drawing_range=[-2.5, 2.5])

    ### ランドマーク ###
    m = Map()
    # for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    ### ゴールの追加 ###
    goal = Goal(0.8, 1.5)
    world.append(goal)

    ### 専有格子地図を追加 ###
    grid_map = GridMap('CorridorGimp_100x100', origin=[-2.5, -2.5, 0])

    ### ロボットを作る ###
    # 初期位置
    # init_pose = np.array([-2.3, 0.5, 0.0])
    init_pose = np.array([-1.0, -0.3, 0.0])
    # 初期位置のばらつき
    # init_pose_stds = np.array([0.1, 0.05, 0.01])
    init_pose_stds = np.array([0.2, 0.02, 0.02])
    # 動作のばらつき
    motion_noise_stds = {"nn":0.02, "no":0.02, "on":0.02, "oo":0.02}

    # 推定器
    estimator = Mcl(m, init_pose, 300, motion_noise_stds=motion_noise_stds,
                    init_pose_stds=init_pose_stds)
    # エージェント
    agent = Pfc(time_interval, 0.1, 0.5, estimator, grid_map, goal, magnitude=2)
    # ロボット
    robot = IdealRobot(init_pose, sensor=Camera(m), agent=agent)
    world.append(robot)

    world.draw()
