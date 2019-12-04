class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def make_decision(self, pose, observation=None):
        return self.nu, self.omega
