import pybullet as p
import pybullet_data
import time

from config import config
from robot import Robot
from camera import Camera

class Simulation:
    def __init__(self):
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        p.loadURDF("plane.urdf")

        self.robot = Robot(config.robot)
        self.camera = Camera(config.camera)

        self.robot_id = self.robot.robot_id
        self.camera_id = self.camera.robot_id

        objects = [vars(obj) for obj in config.objects]

        for i, obj in enumerate(objects):
            start_orientation = p.getQuaternionFromEuler(obj["start_orientation_euler"])
            obj_id = p.loadURDF(
                obj["urdf_path"],
                obj["start_position"],
                start_orientation,
                useFixedBase=False,
                globalScaling=config.global_scaling
            )
            setattr(self, f"object_id_{i+1}", obj_id)

    
    def update(self, attribute_dict: dict):
        self.__dict__.update(attribute_dict)
        

    def step(self):
        p.stepSimulation()