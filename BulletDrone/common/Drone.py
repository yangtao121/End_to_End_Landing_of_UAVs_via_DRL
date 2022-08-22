import numpy as np
import pybullet as p
from BulletDrone.common.DataType import DroneParam, StateVector, SimFlagDrone
from BulletDrone.common.sensor_noise import SensorNoise


class Drone:
    """
    Implementation of drone. You can see as a base class.
    """

    def __init__(self, ID, drone_param: DroneParam, sim_flag: SimFlagDrone):
        self.ID = ID
        self.Ct = drone_param.Ct
        self.Cm = drone_param.Cd
        self.prop_cw = drone_param.prop_cw
        self.prop_pos = drone_param.prop_pos
        self.prop_num = drone_param.prop_num
        self.rot_shape = drone_param.rot_shape
        self.freq_GPS = drone_param.freq_GPS
        self.hover_force = drone_param.mass*9.81
        self.mass = drone_param.mass

        self.inverse_M = np.linalg.inv(drone_param.M)

        self.state = StateVector(drone_param.rot_shape)  # store the state of UAV
        # initial sensor data
        # pos, quat = p.getBasePositionAndOrientation(self.ID)
        # vel, omega = p.getBaseVelocity(self.ID)
        # if self.rot_shape == (3,):
        #     rot = p.getEulerFromQuaternion(quat)
        # elif self.rot_shape == (3, 3):
        #     rot = p.getMatrixFromQuaternion(quat)
        # else:
        #     rot = quat
        # self.state.set_data(pos, vel, rot, omega)

        # sim flag
        # self.sensor_noise_flag = sim_flag.sensor_noise
        if sim_flag.sensor_noise:
            self.sensor_noise = SensorNoise()  # MPU-9250
        else:
            self.sensor_noise = None

        self.sim_dt = sim_flag.dt
        self.GPS_lag = sim_flag.GPS_lag
        self.update_GPS = int(1 / self.freq_GPS / self.sim_dt)

        self.counter = 0  # count the number of runs

    def pwm_input(self, pwm):
        force = pwm ** 2 * self.Ct
        torque = pwm ** 2 * self.Cm
        torque_z = np.sum(torque * (-self.prop_cw))

        # force = 6*np.ones(4)
        # torque_z = 0

        for i in range(self.prop_num):
            p.applyExternalForce(
                self.ID,
                -1,
                [0, 0, force[i]],
                self.prop_pos[i],
                p.LINK_FRAME
            )
            p.setJointMotorControl2(
                bodyUniqueId=self.ID,
                jointIndex=i + 1,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=pwm[i],
                force=1,
            )

        p.applyExternalTorque(
            self.ID,
            -1,
            [0, 0, torque_z],
            p.LINK_FRAME
        )

    def force_tau_input(self, force, tau):
        p.applyExternalForce(
            self.ID,
            -1,
            [0, 0, force],
            np.zeros(3),
            p.LINK_FRAME
        )
        p.applyExternalTorque(
            self.ID,
            -1,
            tau,
            p.LINK_FRAME
        )
        for i in range(self.prop_num):
            p.setJointMotorControl2(
                bodyUniqueId=self.ID,
                jointIndex=i + 1,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=1000,
                force=1,
            )

    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.ID)
        vel, omega = p.getBaseVelocity(self.ID)
        if self.rot_shape == (3,):
            rot = p.getEulerFromQuaternion(quat)
        elif self.rot_shape == (3, 3):
            rot = p.getMatrixFromQuaternion(quat)
        else:
            rot = quat
        pos = np.array(pos)
        rot = np.array(rot)
        vel = np.array(vel)
        omega = np.array(omega)
        if self.sensor_noise:
            pos, vel, rot, omega = self.sensor_noise.add_noise(pos, vel, rot, omega, self.sim_dt)
        if self.GPS_lag:
            if self.counter % self.update_GPS != 0:
                pos = self.state.pos
                vel = self.state.vel
        self.state.set_data(np.array(pos), np.array(vel), np.array(rot), np.array(omega))
        return self.state

    def hover(self):
        force_tau = np.array([self.hover_force, 0, 0, 0])
        self.force_tau_input(force_tau[0], force_tau[1:])

    def get_img(self, width: int = 64, height: int = 64):
        basePos, baseOrientation = p.getBasePositionAndOrientation(self.ID)
        # Obtain the transformation matrix from the quaternion, and obtain the direction (left multiply (1,0,0),
        # because in the original coordinate system, the orientation of the camera is (1,0,0))
        matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=self.ID)
        tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # Transformed X-axis
        tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # Transformed z-axis

        basePos = np.array(basePos)
        # position of camera
        cameraPos = basePos - 0.01 * tz_vec
        targetPos = cameraPos - 0.05 * tz_vec

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cameraPos,
            cameraTargetPosition=targetPos,
            cameraUpVector=tx_vec,
        )
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=70,  # Camera line of sight angle
            aspect=1,
            nearVal=0.06,  # Camera focal length lower limit
            farVal=1000,  # The camera can see the upper limit
        )

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=width, height=height,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            shadow=1
        )

        return width, height, rgbImg, depthImg, segImg

    def set_viewer(self):
        location, orientation = p.getBasePositionAndOrientation(self.ID)
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=110,
            cameraPitch=-30,
            cameraTargetPosition=location,
        )


if __name__ == "__main__":
    drone_param = DroneParam(
        prop_pos=[200, 200, 200],
        inertia=np.ones(3),
        Ct=1,
        Cm=1,
        Cd=1,
        Tm=1,
        mass=1,
        max_rpm=1,
        max_speed=1,
        max_tilt_angle=1,
        freq_GPG=20
    )
    sim_flag = SimFlagDrone(False, False, False)
    print(drone_param.prop_pos)
    drone = Drone(1, drone_param, sim_flag)
    drone.pwm_input(np.ones(4))
