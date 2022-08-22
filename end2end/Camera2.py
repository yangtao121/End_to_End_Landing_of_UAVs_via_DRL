from BulletDrone.common.Drone import Drone
from BulletDrone.control.SimplePID import SimplePIDControl, PIDParam
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.DataType import StateVector, DroneParam, SimFlagDrone
from utils import RewardParameter
import numpy as np
import cv2
import yaml
import copy
import time
import math
import cv2


class Drone_env:
    def __init__(self, drone: Drone, pid_param: PIDParam, drone_parameter: DroneParam, bullet_manager: BulletManager,
                 reward_param: RewardParameter,
                 random_parameter_flag=False,
                 random_pid=False,
                 ):
        self.drone = drone
        self.bullet_manager = bullet_manager
        inverse_M = self.drone.inverse_M
        self.dt = self.bullet_manager.dt
        self.controller = SimplePIDControl(inverse_M, pid_param)

        self.reward_param = reward_param

        # self.interval_choice = np.array([13, 14, 15, 16, 17])
        self.interval_choice = np.array([3])

        self.G = np.array([0, 0, -9.81])
        self.mass = self.drone.mass

        self.state = StateVector()

        dt = 1 / 200.

        omega = 0
        radius = 0
        self.omega = omega * dt

        self.radius = radius

        self.time = 0
        self.random_parameter_flag = random_parameter_flag

        self.drone_param = drone_parameter
        self.random_pid = random_pid

        self.FOV = 70

    def step_attitude(self, angle_fd):
        state = self.drone.get_state()
        fd_tau = self.controller.AttitudeControl(state.rot, state.omega, angle_fd, self.dt)
        # RPM = self.controller.ControlDistribution(fd_tau)
        # self.drone.pwm_input(RPM)
        self.drone.force_tau_input(fd_tau[0], fd_tau[1:])
        self.bullet_manager.stepSimulation()
        return state

    def step_vel(self, vel, yaw=0):
        state = self.drone.get_state()
        euler_fd = self.controller.VelocityControl(state.vel, vel, yaw, self.dt)
        # print(euler_fd)
        fd_tau = self.controller.AttitudeControl(state.rot, state.omega, euler_fd, self.dt)
        # RPM = self.controller.ControlDistribution(fd_tau)
        self.drone.force_tau_input(fd_tau[0], fd_tau[1:])
        self.bullet_manager.stepSimulation()
        return state

    def reset(self):
        to_pos = np.random.uniform(0.4, 1.5, 3)
        sign = np.random.choice(np.array([-1, 1]), 3)
        to_pos = to_pos * sign
        to_pos[2] = np.random.uniform(3, 4, 1)

        if self.random_pid:
            P = np.random.uniform(1, 6, 7)
            I = np.random.uniform(0.01, 1.5, 5)
            D = np.random.uniform(0, 0.3, 5)
            F450_pid_param = PIDParam(
                Kph=P[0],
                Kpz=P[1],
                KVhd=D[0],
                KVhp=P[2],
                KVzp=P[3],
                KVhi=I[0],
                KVzi=I[1],
                KVzd=D[1],
                KOmega_x=20,
                KOmega_y=20,
                KOmega_z=30,
                Kwp_x=P[4],
                Kwp_y=P[5],
                Kwp_z=P[6],
                Kwi_x=I[2],
                Kwi_y=I[3],
                Kwi_z=I[4],
                Kwd_x=D[2],
                Kwd_y=D[3],
                Kwd_z=D[4],
                acc_hmax=0,
                acc_zmax=0,
                delta_hmax=0,
                delta_zmax=0,
                m=self.drone_param.mass,
                inertia=self.drone_param.inertia
            )

            self.controller = SimplePIDControl(self.drone.inverse_M, F450_pid_param)

        if self.random_parameter_flag:
            self.drone_param.random_parameter()
            self.bullet_manager.reset_direct()
            drone_ID = self.bullet_manager.load_drone('src/F450.urdf', to_pos)
            self.bullet_manager.load_background('plane100.urdf')
            self.ID = self.bullet_manager.load_background('src/cube2.urdf')

            F450_pid_param = PIDParam(
                Kph=0,
                Kpz=3,
                KVhd=0,
                KVhp=3,
                KVhi=0,
                KVzp=4,
                KVzi=0,
                KVzd=0,
                KOmega_x=20,
                KOmega_y=20,
                KOmega_z=30,
                Kwp_x=2,
                Kwp_y=2,
                Kwp_z=3,
                Kwi_x=0.8,
                Kwi_y=0.8,
                Kwi_z=2.5,
                Kwd_x=0,
                Kwd_y=0,
                Kwd_z=0,
                acc_hmax=0,
                acc_zmax=0,
                delta_hmax=0,
                delta_zmax=0,
                m=self.drone_param.mass,
                inertia=self.drone_param.inertia
            )

            F450_sim_flag = SimFlagDrone(
                motor_lag=False,
                sensor_noise=False,
                pwm_noise=False,
                GPS_lag=False
            )
            self.drone = Drone(drone_ID, self.drone_param, F450_sim_flag)

            self.controller = SimplePIDControl(self.drone.inverse_M, F450_pid_param)

            self.FOV = np.random.uniform(65, 90)


        # to_pos = np.array([1, 2, 3.5])
        else:
            self.ID = self.bullet_manager.reset(to_pos)
            self.controller.reset()

        width, height, rgbImg, depthImg, segImg = self.drone.get_img(width=64, height=64, FOV=self.FOV)
        img = np.array(rgbImg, dtype=np.float32)
        img = img[:, :, :3]
        # cv2.imwrite('1.jpg', img)
        img = img[:, :, 1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, 64, 64, 1)
        self.time = 0
        # cv2.imwrite('1.jpg', img)
        return img, None

    def shaping(self, state):
        pos = np.linalg.norm(state.pos[:2], ord=2)
        vel = np.linalg.norm(state.vel, ord=2)
        # omega = np.linalg.norm(state.omega, ord=2)
        # shaping = -100 * pos - 10 * vel - 0.1 * state.pos[2] - 5*omega
        shaping_pos_x = -self.reward_param.pos_xy * pos
        shaping_pos_y = - self.reward_param.z * state.pos[2]
        shaping_vel = -self.reward_param.vel * vel
        # - 1.2*state.pos[2]
        return shaping_pos_x, shaping_pos_y, shaping_vel

    def reward2(self, state):
        posx_t_1, posy_t_1, vel_t_1 = self.shaping(self.state)
        posx_t, posy_t, vel_t = self.shaping(state)
        r_posx = posx_t - posx_t_1
        r_posy = posy_t - posy_t_1
        r_vel = vel_t - vel_t_1
        r = r_posx + r_posy + r_vel
        if self.reward_param.special_reward:
            if np.linalg.norm(state.pos[:2]) < 0.2 and state.pos[2] < 0.2:
                r = r + self.reward_param.done_r
            if np.linalg.norm(state.pos[:2]) > 0.6 and state.pos[2] < 0.15:
                r = r + self.reward_param.not_done
        reward_info = np.array([r_posx, r_posy, r_vel])
        return r, reward_info

    def step(self, action):

        x = -self.radius * 1. * self.omega / 2. * math.sin(1. * self.omega * self.time / 2.)
        y = self.radius * self.omega * math.cos(self.omega * self.time)

        self.bullet_manager.engine.resetBaseVelocity(
            self.ID,
            [x, y, 0]
        )
        action = action.numpy()
        action = np.clip(action, -1, 1)
        # acc = action*2
        action[:2] = action[:2] * 0.5
        action[2] = action[2] * 0.8
        # if self.state.pos[2] < 5:
        #     action[2] = action[2] * 0.5
        #     # action[2] = 0.2
        # else:
        #     action[2] = -1

        # action[2] = action[2] * 2

        vel = action
        # vel = np.array([-1, -1, -1])
        # acc[2] = -0.2
        # action = action*0

        yaw = 0
        # target = np.array([acc[0], acc[1]], acc[2], yaw)
        # print(angle_fd)
        interval = np.random.choice(self.interval_choice, 1)[0]
        self.state = copy.deepcopy(self.drone.get_state())
        for i in range(interval):
            state = self.drone.get_state()
            target = self.controller.VelocityControl(state.vel, vel, yaw, interval * self.drone.sim_dt)
            fd_tau = self.controller.AttitudeControl(state.rot, state.omega, target, interval * self.drone.sim_dt)
            self.drone.force_tau_input(fd_tau[0], fd_tau[1:])
            self.drone.set_viewer()
            self.bullet_manager.stepSimulation()

        t = i * 1. / 200
        width, height, rgbImg, depthImg, segImg = self.drone.get_img(width=64, height=64, FOV=self.FOV)

        img = np.array(rgbImg, dtype=np.float32)
        # img = cv2.resize(img, (64,64))
        img = img[:, :, :3]
        img = img[:, :, 1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (64, 64))
        img = img.reshape(1, 64, 64, 1)

        state = self.drone.get_state()
        reward, reward_info = self.reward2(state)

        self.time += 1

        pos, quat = self.bullet_manager.engine.getBasePositionAndOrientation(self.ID)
        vel, omega = self.bullet_manager.engine.getBaseVelocity(self.ID)

        land_pos_vel = np.array([pos, vel])
        drone_pos_vel = np.array([state.pos, state.vel])
        # print(drone_pos_vel)
        # print(pos_vel)
        # reward_info = []
        # time.sleep(0.1)
        return img, reward, land_pos_vel, drone_pos_vel, t

    def reward_parameter(self, path):
        reward_parameter = {
            'pos_xy': self.reward_param.pos_xy,
            'z': self.reward_param.z,
            'vel': self.reward_param.vel,
            'omega': self.reward_param.omega,
            'rot': self.reward_param.rot,
            'special_reward': self.reward_param.special_reward,
            'done_r': self.reward_param.done_r,
            'not_done': self.reward_param.not_done,
            'stop_enable': self.reward_param.stop_enable
        }
        with open(path + '/reward_args.yaml', 'w') as f:
            yaml.dump(reward_parameter, f)

    def transform(self, acc, yaw):
        r3_d = (-self.G + acc) / np.linalg.norm(-self.G + acc)
        pitch_x = np.cos(yaw) * r3_d[0] + np.sin(yaw) * r3_d[1]
        sign_pitch = np.sign(r3_d[2])
        if sign_pitch == 0:
            sign_pitch = 1
        pitch = np.arctan2(sign_pitch * pitch_x, sign_pitch * r3_d[2])
        roll = np.arcsin(np.sin(yaw) * r3_d[0] - np.cos(yaw) * r3_d[1])
        r3_d = r3_d.reshape((1, 3))
        fd = self.mass * r3_d @ (-self.G.reshape((3, 1)) + acc.reshape((3, 1)))
        fd = np.squeeze(fd)
        target = np.array([roll, pitch, yaw, fd])
        # print(target)
        return target
