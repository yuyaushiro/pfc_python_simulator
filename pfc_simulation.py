from modules.world import World, Landmark, Map, Goal


if __name__ == "__main__":
    time_interval = 0.1
    world = World(100, time_interval, drawing_range=[-5, 5])

    m = Map()
    for ln in [(-4,2), (2,-3), (3,3)]: m.append_landmark(Landmark(*ln))
    world.append(m)

    goal = Goal(0.0, 0.0)
    world.append(goal)

    world.draw()
