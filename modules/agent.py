import math


class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def make_decision(self, pose, observation=None):
        return self.nu, self.omega


class EstimationAgent:
    def __init__(self, time_interval, nu, omega, estimator):
        self.nu = nu
        self.omega = omega

        self.estimator = estimator
        self.time_interval = time_interval

        self.prev_nu = 0.0
        self.prev_omega = 0.0

    def make_decision(self, observation=None):
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        return self.nu, self.omega

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)
        x, y, t = self.estimator.pose
        s = "({:.2f}, {:.2f}, {})".format(x,y,int(t*180/math.pi)%360)
        elems.append(ax.text(x, y+0.1, s, fontsize=8))
