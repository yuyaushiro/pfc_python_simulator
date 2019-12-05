import matplotlib
import math
import cv2
import numpy as np


# 格子地図クラス
class GridMap:
    def __init__(self, map_name, resolution=[0.05, 0.05, math.pi/12], origin=[0.0, 0.0, 0.0],
                 obstacle_thresh=254):
        # マップ名
        self.map_name = map_name
        # マップ画像
        self.map_image = cv2.imread('map/' + map_name + '.png', cv2.IMREAD_GRAYSCALE)
        # マップデータ(座標系合わせ後)
        self.map_data = self.map_image.T[:, ::-1]
        # 解像度 (m/pixel)
        self.resolution = np.array(resolution)
        # セルの数
        self.cell_num = np.array([self.map_image.shape[0], self.map_image.shape[1], 24])
        # マップ内の最小の姿勢
        self.pose_min = np.array(origin)
        # マップ内の最大の姿勢
        self.pose_max = self.cell_num*self.resolution + self.pose_min
        # 障害物閾値
        self.obstacle_thresh = obstacle_thresh

        # 価値関数の初期化
        self.value_data = self.init_value(self.cell_num, self.map_name)

    # 価値関数参照
    def value(self, pose):
        index = self.to_index(pose)
        value = self.value_data[index]

        return value

    # 姿勢を整数のインデックスに変換
    def to_index(self, pose):
        index = np.floor((pose - self.pose_min)/self.resolution).astype(int)
        index[2] = (index[2] + self.cell_num[2]*1000)%self.cell_num[2]
        for i in [0, 1]:
            if index[i] < 0: index[i] = 0
            elif index[i] >= self.cell_num[i]: index[i] = self.cell_num[i] - 1

        return tuple(index)

    # 価値観数の初期化
    def init_value(self, cell_num, map_name):
        value_data = np.zeros(np.r_[cell_num])
        for line in open('value/' + map_name + 'x24.value', 'r'):
            d = line.split()
            value_data[int(d[0]), int(d[1]), int(d[2])] = d[3]

        return value_data
