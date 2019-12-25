from modules.robot import IdealRobot
from modules.sensor import IdealCamera
from modules.avoidance import Avoidance

from scipy.stats import multivariate_normal
import numpy as np
import math
import random
import copy


class Particle:
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight
        self.avoidance = Avoidance(1/weight)

    def motion_update(self, nu, omega, time, noise_rate_pdf):
        ns = noise_rate_pdf.rvs()
        pnu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        pomega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = IdealRobot.transition_state(pnu, pomega, time, self.pose)

    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_pos = d[0]
            obs_id = d[1]

            ##パーティクルの位置と地図からランドマークの距離と方角を算出##
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = IdealCamera.calc_relative_position(self.pose, pos_on_map)

            ##尤度の計算##
            distance_dev = distance_dev_rate*particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            self.weight *= multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf(obs_pos)


class Mcl:
    def __init__(self, envmap, init_pose, num,
                 motion_noise_stds={"nn":0.19, "no":0.001, "on":0.13, "oo":0.2},
                 distance_dev_rate=0.14, direction_dev=0.05,
                 init_pose_stds=None):
        if init_pose_stds is None:
            self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        else:
            x_rand = np.random.normal(init_pose[0], init_pose_stds[0], num)
            y_rand = np.random.normal(init_pose[1], init_pose_stds[1], num)
            t_rand = np.random.normal(init_pose[2], init_pose_stds[2], num)
            rand = np.vstack((x_rand, y_rand, t_rand)).T
            self.particles = [Particle(r, 1.0/num) for r in rand]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)
        self.ml = self.particles[0]
        self.pose = self.ml.pose

    def set_ml(self):
        i = np.argmax([p.weight for p in self.particles])
        self.ml = self.particles[i]
        self.pose = self.ml.pose

    def motion_update(self, nu, omega, time):
        for p in self.particles: p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)

    def observation_update(self, observation):
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev)
        self.set_ml()
        self.resampling()

    def resampling(self):
        ws = np.cumsum([e.weight for e in self.particles])
        if ws[-1] < 1e-100: ws = [e + 1e-100 for e in ws]

        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []

        while(len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles: p.weight = 1.0/len(self.particles)

    def draw(self, ax, elems):
        # xs = [p.pose[0] for p in self.particles]
        # ys = [p.pose[1] for p in self.particles]
        # vxs = [math.cos(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        # vys = [math.sin(p.pose[2])*p.weight*len(self.particles) for p in self.particles]
        # elems.append(ax.quiver(xs, ys, vxs, vys, angles='xy', scale_units='xy',
        #                        color="blue", alpha=0.5))
        xs = []
        ys = []
        vxs = []
        vys = []
        for p in self.particles:
            if p.avoidance.weight == p.avoidance.min_weight:
                xs.append(p.pose[0])
                ys.append(p.pose[1])
                vxs.append(math.cos(p.pose[2])*p.weight*len(self.particles))
                vys.append(math.sin(p.pose[2])*p.weight*len(self.particles))
        elems.append(ax.quiver(xs, ys, vxs, vys, angles='xy', scale_units='xy',
                               color="blue", alpha=0.5))
        xs = []
        ys = []
        vxs = []
        vys = []
        for p in self.particles:
            if p.avoidance.weight > p.avoidance.min_weight:
                xs.append(p.pose[0])
                ys.append(p.pose[1])
                vxs.append(math.cos(p.pose[2])*p.weight*len(self.particles))
                vys.append(math.sin(p.pose[2])*p.weight*len(self.particles))
        elems.append(ax.quiver(xs, ys, vxs, vys, angles='xy', scale_units='xy',
                               color="red", alpha=0.5))
