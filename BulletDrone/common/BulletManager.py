import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client as bc


class BulletManager:
    def __init__(self, connection_mode: str, GRAV=-9.81, dt=1 / 200.):
        """
        It can help you manage your bullet engine, like load Drone, load background.
        Notice load background first.
        :param connection_mode: str. 'GUI' or 'DIRECT'
        :param GRAV: gravity.
        """
        self.connection_mode = connection_mode
        if connection_mode == "GUI":
            self.engine = bc.BulletClient(connection_mode=p.GUI)
        elif connection_mode == "DIRECT":
            self.engine = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            raise ValueError("Can not recognize connection_mode")

        self.engine.setAdditionalSearchPath(pybullet_data.getDataPath())

        # set the light direction and render mode.
        self.engine.configureDebugVisualizer(p.ER_TINY_RENDERER, shadowMapIntensity=0, shadowMapWorldSize=10,
                                             lightPosition=[0, 0, 0])
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI)

        self.engine.setTimeStep(dt)

        self.dt = dt
        self.GRAV = GRAV

        self.drone_file = []
        self.drone_pos = []
        self.background_file = []
        self.background_pos = []

    def load_drone(self, file: str, pos=None):
        if pos is None:
            pos = [0, 0, 0]
        self.drone_file.append(file)
        self.drone_pos.append(pos)
        ID = self.engine.loadURDF(file, flags=p.URDF_USE_INERTIA_FROM_FILE, basePosition=pos)

        return ID

    def load_background(self, file: str, pos=None):
        if pos is None:
            pos = [0, 0, 0]

        self.background_file.append(file)
        self.background_pos.append(pos)
        self.engine.loadURDF(file, pos)

    def reset(self, drone_pos=None):
        self.engine.resetSimulation()
        self.engine.setGravity(0, 0, self.GRAV)
        self.engine.setTimeStep(self.dt)
        self.engine.setRealTimeSimulation(0)

        for i, file in enumerate(self.background_file):
            self.engine.loadURDF(file, self.background_pos[i])
        for i, file in enumerate(self.drone_file):
            if drone_pos is None:
                drone_pos = self.drone_pos[i]
            self.engine.loadURDF(file, flags=p.URDF_USE_INERTIA_FROM_FILE, basePosition=drone_pos)

    def reset_direct(self):
        self.engine.disconnect()
        if self.connection_mode == "GUI":
            self.engine = bc.BulletClient(connection_mode=p.GUI)
        elif self.connection_mode == "DIRECT":
            self.engine = bc.BulletClient(connection_mode=p.DIRECT)
        else:
            raise ValueError("Can not recognize connection_mode")
        self.engine.resetSimulation()
        self.engine.setGravity(0, 0, self.GRAV)
        self.engine.setTimeStep(self.dt)
        self.engine.setRealTimeSimulation(0)

    def load_direct(self, file_name, pos=None):
        """
        must use it after reset
        :param pos:
        :param file_name:
        :return:
        """
        if pos is None:
            pos = [0, 0, 0]
        ID = self.engine.loadURDF(file_name, pos)
        return ID

    def stepSimulation(self):
        self.engine.stepSimulation()

    def reset_light(self, shadowMapIntensity, lightPosition: list):
        self.engine.configureDebugVisualizer(p.ER_TINY_RENDERER, shadowMapIntensity=shadowMapIntensity, shadowMapWorldSize=10,
                                             lightPosition=lightPosition)
