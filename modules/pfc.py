from modules.mcl import Particle, Mcl
from modules.grid_map import GridMap
from modules.robot import IdealRobot
from modules.dynamic_programming import DynamicProgramming

import math
import numpy as np
import itertools


class Pfc:
    def __init__(self, time_interval, nu, omega, estimator, grid_map, goal, magnitude=2):
        self.time_interval = time_interval

        # 速度と行動
        self.nu = nu
        self.omega = omega
        self.actions = [(nu, 0.0), (0.0, omega), (0.0, -omega)]

        # 推定用に 1 ステップ前の速度を保存
        self.prev_nu = 0.1
        self.prev_omega = 0.0

        # 推定器
        self.estimator = estimator

        # マップ設定
        self.grid_map = grid_map
        # ゴール設定
        self.goal = goal

        # 積極度
        self.magnitude = magnitude

        self.dp = DynamicProgramming(grid_map.map_image, grid_map.resolution, goal,
                                     0.1, 10, value=grid_map.value_data)
        self.history = [(0, 0)]

    def make_decision(self, pose, observation):
        # 描画用に正しい姿勢を保持
        self.true_pose = pose

        # 状態を推定
        self.estimate_state(self.estimator, observation)

        # 方策計算
        nu, omega = self.calc_policy()

        self.prev_nu, self.prev_omega = nu, omega
        return nu, omega

    def calc_policy(self):
        indexes = [self.grid_map.to_index(p.pose) for p in self.estimator.particles]
        self.evaluations = [self.evaluate(a, indexes) for a in self.actions]

        # self.history.append(self.actions[np.argmax(self.evaluations)])
        # if self.history[-1][0] + self.history[-2][0] == 0.0 and self.history[-1][1] + self.history[-2][1] == 0.0:
        #     return (self.nu, 0.0)
        # return self.history[-1]
        action_index = np.argmax(self.evaluations)

        # 回避重みを決定する
        for p in self.estimator.particles:
            p.avoidance.determine_weight(action_index)

        return self.actions[action_index]

    def evaluate(self, action, indexes):
        v = self.dp.value_function
        # 分母
        vs = [abs(v[i]) if abs(v[i]) > 0.0 else 1e-10 for i in indexes]
        # 分子
        qs = [self.dp.action_value(action, i) for i in indexes]
        avoidance_ms = []
        for (p, q) in zip(self.estimator.particles, qs):
            p.avoidance.weight_candidate_append(q[2])
            print(p.avoidance.weight, p.avoidance.weight_candidates[-1])
            avoidance_ms.append(max([p.avoidance.weight,
                                     p.avoidance.weight_candidates[-1]]))

        q_pfc = sum([( (q[1]-q[2]) * am )\
                     /( v**self.magnitude ) for (v, q, am) in zip(vs, qs, avoidance_ms)])
        return q_pfc

    def estimate_state(self, estimator, observation):
        estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        estimator.observation_update(observation)
        # ゴールに入ったパーティクルを削除
        for p in self.estimator.particles:
            if self.goal.inside(p.pose): p.weight *= 1e-10
        self.estimator.resampling()

        for p in self.estimator.particles:
            p.avoidance.decrease_weight(self.time_interval)

    def draw(self, ax, elems):
        # 推定器の描画
        self.estimator.draw(ax, elems)
