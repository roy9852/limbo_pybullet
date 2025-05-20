import pybullet as p
import numpy as np
import math
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation as R

from config import config

class Camera:
    def __init__(self, CameraConfig = config.camera):
        """
        Initialize the camera.
        
        Args:
            CameraConfig: Configuration for the camera
        """

        # Robot configuration
        self.urdf_path = CameraConfig.urdf_path
        self.ee_index = CameraConfig.ee_index
        self.base_position = CameraConfig.base_position
        self.base_orientation = p.getQuaternionFromEuler(CameraConfig.base_orientation_euler)
        self.p_gain = CameraConfig.p_gain
        self.max_force = CameraConfig.max_force

        # Camera parameters
        self.camera_params = {
            "fov": CameraConfig.fov,
            "aspect": CameraConfig.aspect,
            "near_plane": CameraConfig.near_plane,
            "far_plane": CameraConfig.far_plane,
            "image_width": CameraConfig.image_width,
            "image_height": CameraConfig.image_height
        }

        # Load robot
        self.robot_id = p.loadURDF(self.urdf_path, self.base_position, self.base_orientation, useFixedBase=True)
        
        # Get joint information
        self.controlled_joints = self._get_controllable_joint_indices()
        self.joint_limits = self._get_joint_limits()
        
        # Initialize target and current pose
        self.target_ee_pose = None
        self.target_joint_angles = None
        self.ee_pose = None

        # Move to initial pose
        self.initial_viewpoint = config.robot.initial_position
        self.solve_ik(self.initial_viewpoint)
        self.initial_ee_pose = self.target_ee_pose[:]
        self.initial_joint_angles = self.target_joint_angles[:] 
        for i, joint_index in enumerate(self.controlled_joints):
            p.resetJointState(self.robot_id, joint_index, self.initial_joint_angles[i])


    def _get_controllable_joint_indices(self) -> List[int]:
        """Get indices of controllable joints."""
        controlled_joints = []
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[2] == p.JOINT_REVOLUTE or joint_info[2] == p.JOINT_PRISMATIC:
                controlled_joints.append(i)
        return controlled_joints
    

    def _get_joint_limits(self) -> Tuple[List[float], List[float]]:
        """Get joint limits for all controlled joints."""
        lower_limits = []
        upper_limits = []
        for joint in self.controlled_joints:
            joint_info = p.getJointInfo(self.robot_id, joint)
            lower_limits.append(joint_info[8])
            upper_limits.append(joint_info[9])
        return (lower_limits, upper_limits)
    

    def solve_ik(self, target_viewpoint: List[float]) -> bool:
        """
        Solve inverse kinematics to move the end effector to the target position and orientation.
        
        Args:
            target_viewpoint: Target viewpoint [x, y, z]

        Returns:
            bool: True if IK solution found and applied, False otherwise
        """

        # Step 1: Get current camera pose
        ee_position, ee_orientation = self.get_ee_pose()
        target_position = ee_position

        # Step 2: Calculate forward vector
        forward = np.array(target_viewpoint) - np.array(ee_position)
        forward /= np.linalg.norm(forward)

        # Step 3: Choose temporary up vector that isn't parallel to forward
        tmp_up = np.array([0, 0, 1.0])
        if np.abs(np.dot(forward, tmp_up)) > 0.99:
            tmp_up = np.array([0, 1.0, 0])

        # Step 4: Calculate right and up vectors
        right = np.cross(tmp_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        # Step 5: Calculate rotation matrix 
        rot_matrix = np.column_stack((right, up, forward))
        
        # Step 6: Convert to quaternion (xyzw)
        r = R.from_matrix(rot_matrix)
        quat = r.as_quat()  # returns [x, y, z, w]
        target_orientation = quat.tolist()
        
        # Step 7: Prepare joint constraints
        lower_limits = self.joint_limits[0]
        upper_limits = self.joint_limits[1]
        joint_ranges = [u - l for l, u in zip(lower_limits, upper_limits)]
        rest_poses = [0.0] * len(self.controlled_joints)

        # Step 8: Solve IK
        ik_solution = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_index,
            targetPosition=ee_position,
            targetOrientation=target_orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-5
        )

        if ik_solution is None:
            return False
        else:
            self.target_ee_pose = (target_position, target_orientation)
            self.target_joint_angles = ik_solution
            return True


    def move(self) -> None:
        """Apply joint controls based on IK solution."""
        if self.target_joint_angles is None:
            return
            
        for joint_id, joint_angle in enumerate(self.target_joint_angles):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=self.controlled_joints[joint_id],
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_angle,
                force=self.max_force,
                positionGain=self.p_gain
            )
        self.get_ee_pose()


    def get_ee_pose(self) -> Tuple[List[float], List[float]]:
        """
        Get current end effector pose.
        
        Returns:
            Tuple containing (position, orientation)
        """
        link_state = p.getLinkState(self.robot_id, self.ee_index)
        position = link_state[4]  # worldLinkFramePosition
        orientation = link_state[5]  # worldLinkFrameOrientation
        self.ee_pose = (position, orientation)
        return self.ee_pose
    

    def is_target_reached(self, position_tolerance: float = 0.01, orientation_tolerance: float = 0.1) -> bool:
        """
        Check if current pose is close enough to target pose.
        
        Args:
            position_tolerance: Maximum allowed position error in meters
            orientation_tolerance: Maximum allowed orientation error in radians
            
        Returns:
            bool: True if target is reached, False otherwise
        """
        # Update current pose
        self.get_ee_pose()
            
        # Calculate position error
        pos_error = np.linalg.norm(np.array(self.target_ee_pose[0]) - np.array(self.ee_pose[0]))
        
        # Calculate orientation error
        target_rot = R.from_quat(self.target_ee_pose[1])
        current_rot = R.from_quat(self.ee_pose[1])
        rot_error = (target_rot.inv() * current_rot).magnitude()
        
        return pos_error < position_tolerance and rot_error < orientation_tolerance
    

    def get_rgbd_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get RGBD image from camera.
        
        Returns:
            Tuple containing (RGB image, depth image)
        """

        # Step 1: Get current camera pose
        ee_position, ee_orientation = self.get_ee_pose()

        # Step 2: Convert quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(ee_orientation)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Step 3: Local Z-axis in world frame (camera forward direction)
        forward_vec = rot_matrix[:, 2]  # 3rd column = local +Z
        up_vec = rot_matrix[:, 1]       # 2nd column = local +Y
        
        # Step 4: Shift camera target slightly along forward direction
        cam_target = [ee_position[i] + forward_vec[i] for i in range(3)]

        # Step 5: Compute view and projection matrices
        view_matrix = p.computeViewMatrix(ee_position, cam_target, up_vec)
        proj_matrix = p.computeProjectionMatrixFOV(self.camera_params["fov"], self.camera_params["aspect"], self.camera_params["near_plane"], self.camera_params["far_plane"])

        # Step 6: Capture image
        images = p.getCameraImage(self.camera_params["image_width"], self.camera_params["image_height"], view_matrix, proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.reshape(images[2], (self.camera_params["image_height"], self.camera_params["image_width"], 4))[:, :, :3]
        depth_buffer = np.reshape(images[3], [self.camera_params["image_height"], self.camera_params["image_width"]])

        # Step 7: Convert depth buffer to real depth values
        depth_array = self.camera_params["far_plane"] * self.camera_params["near_plane"] / (self.camera_params["far_plane"] - (self.camera_params["far_plane"] - self.camera_params["near_plane"]) * depth_buffer)

        return (rgb_array, depth_array)
