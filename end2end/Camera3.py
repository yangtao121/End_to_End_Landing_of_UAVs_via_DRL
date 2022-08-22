from BulletDrone.common.Drone import Drone
from BulletDrone.control.SimplePID import SimplePIDControl, PIDParam
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.DataType import StateVector
from utils import RewardParameter
import numpy as np
import cv2
import yaml
import copy
import time
import math
import cv2
from BulletDrone.utils.FileTools import copy_file
from BulletDrone.utils.URDFTools import Link, URDF
import os

body = """
  newmtl Material
  Ns 10.0000
  Ni 1.5000
  d 1.0000
  Tr 0.0000
  Tf 1.0000 1.0000 1.0000 
  illum 2
  Ka 0.0000 0.0000 0.0000
  Kd 0.5880 0.5880 0.5880
  Ks 0.0000 0.0000 0.0000
  Ke 0.0000 0.0000 0.0000

"""


def generate_mtl(name, map_kd):
    file = body + "  map_Kd " + map_kd
    with open(name, "w") as f:
        f.write(file)



class Drone_env:
    def __init__(self, drone: Drone, pid_param: PIDParam, bullet_manager: BulletManager, reward_param: RewardParameter):
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
        radius = 200
        self.omega = omega * dt

        self.radius = radius

        self.time = 0

        self.file_start = 0
        self.file_end = 120

        self.workspace = 'workspace/' + str(1)

        self.counter = 0

        self.random_pictures = None
        self.land_color_flag = False
        self.land_size_flag = True
        self.noise_land_flag = True

        self.land_or = 1.5
        self.init_random_pos = 1.2

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
        self.counter += 1
        self.random_floor()
        to_pos = np.random.uniform(0., 0.2, 3)
        sign = np.random.choice(np.array([-1, 1]), 3)
        to_pos = to_pos * sign
        to_pos[2] = np.random.uniform(1.0, 2.0, 1)

        self.bullet_manager.reset_direct()

        self.bullet_manager.load_direct(self.workspace + '/' + 'plane{}.urdf'.format(self.counter))
        self.bullet_manager.load_direct(self.workspace + '/' + 'landing.urdf')

        self.ID = self.bullet_manager.load_direct('src/F450.urdf', to_pos)
        self.drone.ID = self.ID

        self.random_floor_noise()

        # to_pos = np.array([1, 2, 3.5])

        self.controller.reset()
        width, height, rgbImg, depthImg, segImg = self.drone.get_img(width=720, height=480)
        img = np.array(rgbImg, dtype=np.float32)
        # cv2.imwrite('1.jpg', img)
        img = img[:, :, :3]
        img = img[:, :, ::-1]
        cv2.imwrite('1.jpg', img)

        img = img[:, :, 1]

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape(1, 64, 64, 1)
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

    def random_floor(self):
        tag = np.random.uniform(self.file_start, self.file_end, 1)
        tag = round(float(tag))
        if self.random_pictures == 1:
            copy_file('source', self.workspace, 'base.png', 'texture{}.png'.format(self.counter))

        else:
            copy_file('source/texture/', self.workspace, str(tag) + '.png', 'texture{}.png'.format(self.counter))

        generate_mtl(self.workspace + "/plane.mtl", "texture{}.png".format(self.counter))

        # random direction and size about landing
        landing_urdf = URDF("landing area")
        landing_link = Link('baseLink')
        origin = np.zeros(6)
        origin_0 = origin.tolist()
        origin[5] = np.random.uniform(-self.land_or, self.land_or, 1)[0]
        origin = origin.tolist()
        landing_link.inertial(origin_0, origin_0, 0)
        if self.land_color_flag:
            color = np.random.uniform(0, 1, 4)
        else:
            color = np.array([1, 0, 0, 0])
        color[3] = 1
        color = color.tolist()
        if self.land_size_flag:
            size = np.random.uniform(0.4, 0.6, 3)
        else:
            size = np.array([0.4, 0.4, 0])
        size[2] = 0.01
        size = size.tolist()
        landing_link.visual(origin, color, size)
        landing_urdf.add_link(landing_link)
        landing_urdf.save(self.workspace + '/landing.urdf')

        # random texture
        plane_urdf = URDF("plane")
        plane_link = Link("baseLink")
        plane_link.contact(1)
        plane_link.inertial(origin_0, origin_0, 0)
        color = np.ones(4)
        color.tolist()
        size = np.ones(3)
        size = size.tolist()
        plane_link.visual(origin_0, color, size, "plane{}.obj".format(self.counter))
        origin_0 = np.zeros(6)
        origin_0[2] = -5
        plane_link.collision(origin_0, [100, 100, 10])
        plane_urdf.add_link(plane_link)
        plane_urdf.save(self.workspace + "/plane{}.urdf".format(self.counter))
        copy_file('source', self.workspace, "plane.obj", "plane{}.obj".format(self.counter))
        tag = int(self.counter - 1)
        # print(tag)
        try:
            os.remove(self.workspace + "/plane{}.urdf".format(tag))
            os.remove(self.workspace + "/plane{}.obj".format(tag))
        except:
            # raise ValueError('not exist')
            pass
        if tag:
            try:
                os.remove(self.workspace + "/texture{}.png".format(tag))
            except:
                # raise ValueError("not exist")
                pass

    def step(self, action):

        x = -self.radius * 1.2 * self.omega * math.sin(1.2 * self.omega * self.time)
        y = self.radius * self.omega * math.cos(self.omega * self.time)

        self.bullet_manager.engine.resetBaseVelocity(
            self.ID,
            [x, y, 0]
        )
        action = action.numpy()
        action = np.clip(action, -1, 1)
        # acc = action*2
        action[:2] = action[:2] * 1
        # action[2] = action[2] * 2
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
        width, height, rgbImg, depthImg, segImg = self.drone.get_img(width=64, height=64)

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

    def random_floor_noise(self):
        # adding Interference items
        num = np.random.uniform(30, 100)
        noise_items_num = int(num)
        for i in range(noise_items_num):
            shape = np.random.uniform(0, 100)
            # print(int(shape) % 2)
            if int(shape) % 2:
                size = np.random.uniform(0.01, 0.15, 3)
                size[2] = 0.01
            else:
                size = np.random.uniform(0.01, 0.1, 2)
                size[1] = 0.01

            size = size.tolist()
            origin = np.zeros(6)
            origin_0 = origin.tolist()
            origin[5] = np.random.uniform(-1.5, 1.5, 1)[0]
            origin = origin.tolist()
            color = np.random.uniform(0, 1, 4)
            color[3] = 1
            color = color.tolist()

            urdf = URDF("Interference")
            link = Link("baseLink")
            link.inertial(origin_0, origin_0, 0)
            link.visual(origin, color, size)
            urdf.add_link(link)

            file = self.workspace + '/noise/noise_' + str(i) + '.urdf'

            urdf.save(file)
            pos = np.random.uniform(0.01, 1.5, 3)
            sign = np.random.choice([-1, 1])
            pos = sign * pos
            pos[2] = 0
            self.bullet_manager.load_direct(file, pos)

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
