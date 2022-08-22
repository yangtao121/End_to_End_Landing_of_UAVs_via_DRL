import numpy as np


class DroneParam:
    """
    UAV parameter manager
    """

    def __init__(self, prop_pos, inertia, Ct, Cm, Cd, Tm, mass, max_rpm, max_speed, max_tilt_angle, freq_GPG,
                 prop_num=4,
                 prop_cw=np.array([1, -1, 1, -1]), rot_shape=(3,)):
        """

        :param prop_pos: Position of propeller。numpy array.
        :param Ct: Propeller Integrated Thrust Coef. float.
        :param Cm: Propeller Integrated Moment Coef. float.
        :param Cd: Air-Drag Coef. float.
        :param Tm: Motor Response Time Constant. float.
        :param mass: UAV's mass. float.
        :param max_rpm: Max motor speed(rpm). float.
        :param max_speed: m/s.
        :param max_tilt_angle: rad/s.
        :param freq_GPG: GPS update frequency. HZ.
        :param prop_num: Number of propeller. int.
        :param prop_cw: Motor rotation direction. numpy array dtype: np.int. 1 indicates clockwise.
        :param rot_shape: rotation shape.
                value:
                    value:
                    (3,3): rotation matrix.
                    (3,): Euler angle.
                    (4,): quaternion.
        """
        x_sign = np.array([1, -1, -1, 1])
        y_sign = np.array([-1, -1, 1, 1])
        self.prop_pos = np.vstack((prop_pos[0] * x_sign, prop_pos[1] * y_sign, prop_pos[2] * np.ones(4)))
        self.inertia = inertia
        self.prop_pos = self.prop_pos.T
        self.Ct = Ct
        self.Cm = Cm
        self.Cd = Cd
        self.Tm = Tm
        self.mass = mass
        self.max_rpm = max_rpm
        self.max_speed = max_speed
        self.max_tilt_angle = max_tilt_angle
        self.freq_GPS = freq_GPG
        self.prop_num = prop_num
        self.prop_cw = prop_cw
        self.prop_num = prop_num
        self.rot_shape = rot_shape

        self.M = np.array(
            [
                [Ct, Ct, Ct, Ct],
                [-Ct * prop_pos[1], -Ct * prop_pos[1], Ct * prop_pos[1], Ct * prop_pos[1]],
                [-Ct * prop_pos[0], Ct * prop_pos[0], Ct * prop_pos[0], -Ct * prop_pos[0]],
                -prop_cw * Cm
            ]
        )

        # print(self.M)


class StateVector:
    """
    Store the status of the UAV.
    """

    def __init__(self, rot_shape=(3,)):
        """

        :param rot_shape: rotation.
                value:
                    (3,3): rotation matrix.
                    (3,): Euler angle.
                    (4,): quaternion.
        """
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        # self.rot represents the rotation
        if rot_shape == (3, 3):
            self.rot = np.eye(3)
        else:
            self.rot = np.zeros(rot_shape)
        self.omega = np.zeros(3)  # quaternion

    def set_data(self, pos, vel, rot, omega):
        self.pos = pos
        self.vel = vel
        self.rot = rot
        self.omega = omega


class SimFlagDrone:
    """
    How to simulate.
    """

    def __init__(self, motor_lag, sensor_noise, pwm_noise, GPS_lag, dt=0.005):
        self.motor_lag = motor_lag
        self.sensor_noise = sensor_noise
        self.pwm_noise = pwm_noise
        self.GPS_lag = GPS_lag
        self.dt = dt


if __name__ == "__main__":
    param = DroneParam([200, 200, 255], Ct=1, Cm=1, Cd=1, inertia=np.ones(3), Tm=1, mass=1, max_rpm=1, max_speed=1,
                       max_tilt_angle=1,
                       freq_GPG=10)
    # F = np.array([1, 1, 0])
    # matrix = np.cross(F, param.prop_pos)
    # a = 100
    # prop_pos = np.array([[a, -a, 0],
    #                      [a, a, 0],
    #                      [-a, a, 0],
    #                      [-a, -a, 0]
    #                      ])
    # print(np.cross(F, prop_pos))
    # print(param.M)
    # file = '123'
    # files = []
    # for file_file in files:
    #     print("ok")
    # files.append(file)
    # files.append(file)
    # print(files)
    # a = np.array([13, 14, 15, 16, 17])
    # b = np.random.choice(a, 1)
    c = np.arctan2(-1, 1)

    print(c)
