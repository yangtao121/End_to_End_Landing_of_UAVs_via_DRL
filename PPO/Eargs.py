class RewardParameter:
    def __init__(self, pos_xy, z, vel, acc, omega, rot, special_reward=False, done_r=5, not_done=-2, stop_enable=True):
        self.pos_xy = pos_xy
        self.z = z
        self.vel = vel
        self.acc = acc
        self.omega = omega
        self.rot = rot
        self.special_reward = special_reward
        self.done_r = done_r
        self.not_done = not_done
        self.stop_enable = stop_enable
