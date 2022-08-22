from BulletDrone.common.Drone import Drone
from BulletDrone.control.SimplePID import SimplePIDControl, PIDParam
from BulletDrone.common.BulletManager import BulletManager


class Drone_env:
    def __init__(self, drone: Drone, pid_param: PIDParam, bullet_manager: BulletManager):
        self.drone = drone
        self.bullet_manager = bullet_manager
        inverse_M = self.drone.inverse_M
        self.dt = self.bullet_manager.dt
        self.controller = SimplePIDControl(inverse_M, pid_param)

    def step_pos(self, pos, yaw=0):
        state = self.drone.get_state()
        euler_fd = self.controller.PositionControl(state.pos, state.vel, pos, yaw, self.dt)
        # print(euler_fd)
        fd_tau = self.controller.AttitudeControl(state.rot, state.omega, euler_fd, self.dt)
        # RPM = self.controller.ControlDistribution(fd_tau)
        self.drone.force_tau_input(fd_tau[0], fd_tau[1:])
        self.bullet_manager.stepSimulation()
        return state

    def reset(self):
        self.bullet_manager.reset()
        state = self.drone.get_state()
        return state

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
