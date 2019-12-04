from modules.world import World


if __name__ == "__main__":
    time_interval = 0.1

    world = World(100, time_interval, drawing_range=[-5, 5])

    world.draw()
