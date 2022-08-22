from BulletDrone.env.SimplePIDEnv import Drone_env
from BulletDrone.common.Drone import Drone
from BulletDrone.common.BulletManager import BulletManager
from BulletDrone.common.DataType import DroneParam, SimFlagDrone
from BulletDrone.control.SimplePID import PIDParam
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
    sensor_noise=True,
    pwm_noise=False,
    GPS_lag=False
)

GUI_flag = False
if GUI_flag:
    bullet_manager = BulletManager(
        connection_mode='GUI',
    )
else:
    bullet_manager = BulletManager(
        connection_mode='DIRECT',
    )

bullet_manager.load_background('plane100.urdf')
bullet_manager.load_background('src/cube2.urdf')
F45_id = bullet_manager.load_drone('src/F450.urdf')

F450_drone = Drone(F45_id, F450_param, F450_sim_flag)

F450_pid_param = PIDParam(
    Kph=0,
    Kpz=0,
    KVhd=0,
    KVhp=0,
    KVhi=0,
    KVzp=0,
    KVzi=0,
    KVzd=0,
    KOmega_x=2,
    KOmega_y=2,
    KOmega_z=3,
    Kwp_x=2,
    Kwp_y=2,
    Kwp_z=3,
    Kwi_x=2,
    Kwi_y=2,
    Kwi_z=3,
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
target = [0, 0, 1.3, 20]
step = 800
angle = np.zeros((step, 3))
for i in range(step):
    state = env.step_attitude(target)
    env.drone.set_viewer()
    angle[i] = state.rot
    if GUI_flag:
        time.sleep(0.02)

# print(angle)
plt.subplot(3, 1, 1)
plt.plot(angle[:, 0])
plt.xlabel('roll')
plt.ylabel('rad')
plt.subplot(3, 1, 2)
plt.plot(angle[:, 1])
plt.xlabel('pitch')
plt.ylabel('rad')
plt.subplot(3, 1, 3)
plt.plot(angle[:, 2])
plt.xlabel('yaw')
plt.ylabel('rad')
plt.show()
