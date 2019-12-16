import numpy as np
import math
import itertools
import collections
from copy import copy
import cv2
import seaborn as sns


class DynamicProgramming:
    def __init__(self, map_image, widths, goal, time_interval, sampling_num, value=None,
                 nu=0.1, omega=0.5):
        # マップ牙城のy軸を反転
        self.map_image = map_image.T[:, ::-1]
        # ピクセル数
        x_pixel, y_pixel = map_image.shape
        nt = int(math.pi*2/widths[2])
        self.index_nums = np.array([x_pixel, y_pixel, nt])
        self.indexes = list(itertools.product(range(x_pixel), range(y_pixel), range(nt)))

        self.pose_min = np.array([0, 0, 0])
        self.pose_max = np.array([x_pixel*widths[0], y_pixel*widths[1], math.pi*2])

        self.widths = widths
        self.goal = goal

        self.value_function, self.final_state_flags, self.obstacle_state_flags =\
                self.init_value_function()
        self.policy = self.init_policy()
        self.actions = [(0.0, omega), (nu, 0.0), (0.0, -omega)]

        self.state_transition_probs = self.init_state_transition_probs(time_interval, sampling_num)

        self.time_interval = time_interval

        if value is not None:
            self.value_function = value

    def value_iteration_sweep(self):
        max_delta = 0.0
        for index in self.indexes:
            if self.final_state_flags[index]:
                self.value_function[index] = 0.0
            else:
                max_q = -1e100
                qs = [self.action_value(a, index) for a in self.actions]
                max_q = max(qs)
                max_a = self.actions[np.argmax(qs)]

                delta = abs(self.value_function[index] - max_q)
                max_delta = delta if delta > max_delta else max_delta

                self.value_function[index] = max_q
                self.policy[index] = np.array(max_a).T

        return max_delta

    def action_value(self, action, index): #はみ出しペナルティー追加
        value_n = 0.0
        reward = 0.0
        for delta, prob in self.state_transition_probs[(action, index[2])]:
            after, edge_reward = self.edge_correction(np.array(index).T + delta)
            after = tuple(after)
            reward += -(self.time_interval\
                        +(self.map_image[after[0], after[1]]<10)*100*self.time_interval\
                        -edge_reward) * prob
            value_n += self.value_function[after] * prob

        action_value = value_n + reward

        return action_value, value_n, reward

    def edge_correction(self, index): #変更
        edge_reward = 0.0
        index[2] = (index[2] + self.index_nums[2])%self.index_nums[2] #方角の処理

        for i in range(2):
            if index[i] < 0:
                index[i] = 0
                edge_reward = -1e100
            elif index[i] >= self.index_nums[i]:
                index[i] = self.index_nums[i]-1
                edge_reward = -1e100

        return index, edge_reward

    def init_policy(self):
        tmp = np.zeros(np.r_[self.index_nums, 2])
        return tmp

    def init_state_transition_probs(self, time_interval, sampling_num):
        ###セルの中の座標を均等にsampling_num**3点サンプリング###
        dx = np.linspace(0.00001, self.widths[0]*0.99999, sampling_num)
        dy = np.linspace(0.00001, self.widths[1]*0.99999, sampling_num)
        dt = np.linspace(0.00001, self.widths[2]*0.99999, sampling_num)
        samples = list(itertools.product(dx, dy, dt))

        ###各行動、各方角でサンプリングした点を移動してインデックスの増分を記録###
        tmp = {}
        for a in self.actions:
            for i_t in range(self.index_nums[2]):
                transitions = []
                for s in samples:
                    before = np.array([s[0], s[1], s[2] + i_t*self.widths[2]]).T + self.pose_min
                    before_index = np.array([0, 0, i_t]).T                                                      #遷移前のインデックス

                    after = self.transition_state(a[0], a[1], time_interval, before)
                    after_index = np.floor((after - self.pose_min)/self.widths).astype(int)

                    transitions.append(after_index - before_index)

                unique, count = np.unique(transitions, axis=0, return_counts=True)
                probs = [c/sampling_num**3 for c in  count]
                tmp[a, i_t] = list(zip(unique, probs))

        return tmp

    def init_value_function(self):
        v = np.empty(self.index_nums)
        f = np.zeros(self.index_nums)
        o = np.empty(self.index_nums)

        for index in self.indexes:
            f[index] = self.final_state(np.array(index).T)
            o[index] = True if self.map_image[index[0], index[1]] < 255 else False
            v[index] = 0.0 if f[index] else - 1000.0

        return v, f, o

    def final_state(self, index):
        x_min, y_min, _ = self.pose_min + self.widths*index
        x_max, y_max, _ = self.pose_min + self.widths*(index + 1)

        corners = [[x_min, y_min, _], [x_min, y_max, _], [x_max, y_min, _], [x_max, y_max, _] ] #4隅の座標
        return all([self.goal.inside(np.array(c).T) for c in corners ])

    def transition_state(self, nu, omega, time, pose):
        t0 = pose[2]
        if math.fabs(omega) < 1e-10:
            return pose + np.array( [nu*math.cos(t0),
                                     nu*math.sin(t0),
                                     omega ] ) * time
        else:
            return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                                     nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                                     omega*time ] )
