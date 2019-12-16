class Avoidance:
    def __init__(self):
        self.weight = 1.0
        self.time = 3.0
        self.elapsed_time = 5.0
        self.max_weight = 100

    def increase_weight(self, time_interval):
        self.elapsed_time = 0.0
        self.weight = 100

    def decrease_weight(self, time_interval):
        if self.weight != 1.0:
            if self.elapsed_time < self.time:
                self.elapsed_time += time_interval
            else:
                self.weight -= self.max_weight/5 * time_interval
                if self.weight < 1.0:
                    self.weight = 1.0
