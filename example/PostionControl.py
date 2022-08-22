from BulletDrone.env.SimplePIDEnv import Drone_env
from BulletDrone.common.Drone import Drone
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.DataType import DroneParam, SimFlagDrone
from BulletDrone.control.SimplePID import PIDParam
import pybullet as p
import time
import numpy as np
from matplotlib import pyplot as plt

F450_param = DroneParam(
    prop_pos=np.array([0.1597, 0.1597, 0]),
    inertia=np.array([1.745e-2, 1.745e-2, 3.175e-2]),
    Ct=1.105e-5,
    Cm=1.489e-7,
    Cd=6.579e-2,
    Tm=0.0136,
    mass=1.5,
    max_rpm=7408.7,
    max_speed=13.4,
    max_tilt_angle=0.805,
    freq_GPG=10
)

F450_sim_flag = SimFlagDrone(
    motor_lag=False,
    sensor_noise=False,
    pwm_noise=False,
    GPS_lag=False
)

bullet_manager = BulletManager(
    connection_mode='DIRECT',
)

bullet_manager.load_background('plane100.urdf')
bullet_manager.load_background('src/cube2.urdf')
F45_id = bullet_manager.load_drone('src/F450.urdf', [0, 0, 0])
info = p.getDynamicsInfo(F45_id, -1)
print(info)

F450_drone = Drone(F45_id, F450_param, F450_sim_flag)

F450_pid_param = PIDParam(
    Kph=0,
    Kpz=4,
    KVhd=0,
    KVhp=0,
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
    m=F450_param.mass,
    inertia=F450_param.inertia
)

env = Drone_env(
    drone=F450_drone,
    pid_param=F450_pid_param,
    bullet_manager=bullet_manager
)
env.reset()
target = [0, 0, 2]
step = 1600
angle = np.zeros((step, 3))
pos = np.zeros((step, 3))
for i in range(step):
    # if i > 800:
    #     target = np.array([0, 0, 1])
    state = env.step_pos(target, 0.5)
    # env.drone.hover()
    # env.bullet_manager.stepSimulation()
    env.drone.set_viewer()
    #     angle[i] = state.rot
    pos[i] = state.pos
    # time.sleep(0.01)
#
# print(angle)
plt.subplot(3, 1, 1)
plt.plot(pos[:, 0])
plt.subplot(3, 1, 2)
plt.plot(pos[:, 1])
plt.subplot(3, 1, 3)
plt.plot(pos[:, 2])
plt.show()
