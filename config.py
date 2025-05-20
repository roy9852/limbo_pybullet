from dataclasses import dataclass
from typing import List
import math
import os

@dataclass
class RobotConfig:
    # Initialization
    initial_position: List[float]
    initial_vector: List[float]
    initial_twist: float
    initial_close: bool
    # Pose and control
    urdf_path: str
    ee_index: int
    base_position: List[float]
    base_orientation_euler: List[float]
    p_gain: float
    max_force: float

@dataclass
class CameraConfig:
    # Intrinsics
    fov: float
    aspect: float
    near_plane: float
    far_plane: float
    image_width: int
    image_height: int
    # Pose and control
    urdf_path: str
    ee_index: int
    base_position: List[float]
    base_orientation_euler: List[float]
    p_gain: float
    max_force: float

@dataclass
class ObjectConfig:
    urdf_path: str
    start_position: List[float]
    start_orientation_euler: List[float]

@dataclass
class SimulationConfig:
    global_scaling: float
    robot: RobotConfig
    camera: CameraConfig
    objects: List[ObjectConfig]

# Initialize configuration with multiple objects
config = SimulationConfig(
    global_scaling=0.08,
    robot=RobotConfig(
        initial_position=[0.0, 0.0, 0.3],
        initial_vector=[0.0, 0.0, -1.0],
        initial_twist=0.0,
        initial_close = False,
        # urdf_path=os.path.join(os.getcwd(), "robots/iiwa14", "iiwa14.urdf"),
        urdf_path=os.path.join(os.getcwd(), "franka_robot", "panda.urdf"),
        ee_index=8,
        base_position=[0.5, 0.0, 0.8],
        base_orientation_euler=[0.0, math.pi, 0.0],
        p_gain=0.1,
        max_force=1000.0
    ),
    camera=CameraConfig(
        fov=60.0,
        aspect=1.0,
        near_plane=0.01,
        far_plane=100.0,
        image_width=512,
        image_height=512,
        urdf_path=os.path.join(os.getcwd(), "robots/iiwa14", "camera.urdf"),
        ee_index=2,
        base_position=[-1.0, 0.0, 1.0],
        base_orientation_euler=[0.0, math.pi, 0.0],
        p_gain=0.1,
        max_force=50.0
    ),
    objects=[
        ObjectConfig(
            urdf_path="ycb_assets/011_banana.urdf",
            start_position=[0.0, 0.4, 0.05],
            start_orientation_euler=[0.0, 0.0, 0.0]
        ),
        ObjectConfig(
            urdf_path="ycb_assets/013_apple.urdf",
            start_position=[0.0, -0.4, 0.05],
            start_orientation_euler=[0.0, 0.0, 0.0]
        )
    ]
)
