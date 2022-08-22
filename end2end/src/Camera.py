from BulletDrone.common.Drone import Drone
from BulletDrone.control.SimplePID import SimplePIDControl, PIDParam
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.DataType import StateVector
from args import RewardParameter
import numpy as np
import cv2
import yaml
import copy
import time


class Drone_env:
    def __init__(self, drone: Drone, pid_param: PIDParam, bullet_manager: BulletManager, reward_param: RewardParameter):
        self.drone = drone
        self.bullet_manager = bullet_manager
        inverse_M = self.drone.inverse_M
        self.dt = self.bullet_manager.dt
        self.controller = SimplePIDControl(inverse_M, pid_param)

        self.reward_param = reward_param

        self.interval_choice = np.array([13, 14, 15, 16, 17])

        self.G = np.array([0, 0, -9.81])
        self.mass = self.drone.mass

        self.state = StateVector()

    def step_attitude(self, angle_fd):
        state = self.drone.get_state()
        fd_tau = self.controller.AttitudeControl(state.rot, state.omega, angle_fd, self.dt)
        # RPM = self.controller.ControlDistribution(fd_tau)
        # self.drone.pwm_input(RPM)
        self.drone.force_tau_input(fd_tau[0], fd_tau[1:])
        self.bullet_manager.stepSimulation()
        return state

    def reset(self):
        to_pos = np.random.uniform(0.2, 0.4, 3)
        sign = np.random.choice(np.array([-1, 1]), 3)
        to_pos = to_pos * sign
        to_pos[2] = np.random.uniform(3.5, 4, 1)
        self.bullet_manager.reset(to_pos)
        self.controller.reset()
        width, height, rgbImg, depthImg, segImg = self.drone.get_img()
        img = np.array(rgbImg, dtype=np.float32)
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, 64, 64, 1)
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
            if np.linalg.norm(state.pos[:2]) < 0.2 and state.pos[2] < 0.3:
                r = r + self.reward_param.done_r
            if np.linalg.norm(state.pos[:2]) > 0.6 and state.pos[2] < 0.3:
                r = r + self.reward_param.not_done
        reward_info = np.array([r_posx, r_posy, r_vel])
        return r, reward_info

    def step(self, action):
        action = action.numpy()
        action = np.clip(action, -1, 1)
        # acc = action*2
        acc = action * np.array([2, 2, 0.5])
        # acc = np.array([0.1, 0.1, -0.01])
        # acc[2] = -0.2
        # action = action*0

        yaw = 0
        target = self.transform(acc, yaw)
        # print(angle_fd)
        interval = np.random.choice(self.interval_choice, 1)[0]
        self.state = copy.deepcopy(self.drone.get_state())
        for i in range(interval):
            state = self.drone.get_state()
            fd_tau = self.controller.AttitudeControl(state.rot, state.omega, target, self.dt)
            self.drone.force_tau_input(fd_tau[0], fd_tau[1:])
            self.drone.set_viewer()
            self.bullet_manager.stepSimulation()

        width, height, rgbImg, depthImg, segImg = self.drone.get_img()
        img = np.array(rgbImg, dtype=np.float32)
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64, 64))
        img = img.reshape(1, 64, 64, 1)

        state = self.drone.get_state()
        reward, reward_info = self.reward2(state)
        # reward_info = []
        # time.sleep(0.1)
        return img, reward, None, reward_info

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
