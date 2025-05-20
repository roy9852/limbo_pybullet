import pybullet as p
import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial.transform import Rotation as R

from config import config

class Robot:
    def __init__(self, RobotConfig = config.robot):
        """
        Initialize the robot.
        
        Args:
            RobotConfig: Configuration for the robot
        """

        self.urdf_path = RobotConfig.urdf_path
        self.ee_index = RobotConfig.ee_index
        self.base_position = RobotConfig.base_position
        self.base_orientation = p.getQuaternionFromEuler(RobotConfig.base_orientation_euler)
        self.p_gain = RobotConfig.p_gain
        self.max_force = RobotConfig.max_force

        # Load robot
        self.robot_id = p.loadURDF(self.urdf_path, self.base_position, self.base_orientation, useFixedBase=True)
        
        # Get joint information
        self.controlled_joints = self._get_controllable_joint_indices()
        self.joint_limits = self._get_joint_limits()
        
        # Gripper
        self.close = RobotConfig.initial_close

        # Initialize target and current pose
        self.target_ee_pose = None
        self.target_joint_angles = None
        self.ee_pose = None
        self.joint_angles = None
        self.initial_joint_angles = None
        self.initial_ee_pose = None

        # Move to initial pose
        self.initial_position = RobotConfig.initial_position
        self.initial_vector = RobotConfig.initial_vector
        self.initial_twist = RobotConfig.initial_twist
        self.solve_ik(self.initial_position, self.initial_vector, self.initial_twist)
        self.initial_ee_pose = self.target_ee_pose[:]
        self.initial_joint_angles = self.target_joint_angles[:] 
        for i, joint_index in enumerate(self.controlled_joints):
            p.resetJointState(self.robot_id, joint_index, self.initial_joint_angles[i])
        self.get_ee_pose()
        self.get_joint_angles()


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
    

    def solve_ik(self, target_position: List[float], target_vector: List[float], target_twist: float) -> bool:
        """
        Solve inverse kinematics to move the end effector to the target position and orientation.
        
        Args:
            target_position: Target position [x, y, z]
            target_vector: Direction the end-effector should point to (e.g., [0, 0, -1])
            target_twist: Rotation around the pointing axis in radians

        Returns:
            bool: True if IK solution found and applied, False otherwise
        """

        # Step 1: Normalize the desired pointing direction
        target_vector = np.array(target_vector)/np.linalg.norm(target_vector)

        # Step 2: Define the original local forward axis (z-axis for most end-effectors)
        original_axis = np.array([0, 0, 1])

        # Step 3: Compute alignment rotation
        align_rot = R.align_vectors([target_vector], [original_axis])[0]

        # Step 4: Apply twist around new z-axis
        local_z_axis = align_rot.apply(original_axis)
        twist_rot = R.from_rotvec(local_z_axis * target_twist)

        # Step 5: Combine alignment + twist, convert to quaternion
        final_rot = twist_rot * align_rot
        target_orientation = final_rot.as_quat()  # [x, y, z, w]

        # Step 6: Prepare joint constraints
        lower_limits = self.joint_limits[0]
        upper_limits = self.joint_limits[1]
        joint_ranges = [u - l for l, u in zip(lower_limits, upper_limits)]
        # if self.initial_joint_angles is None:
        #     rest_poses = [0.0] * len(self.controlled_joints)
        # else:
        #     rest_poses = self.initial_joint_angles
        rest_poses = [0.0] * len(self.controlled_joints)

        # Step 7: Solve IK
        ik_solution = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_index,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=1500,
            residualThreshold=1e-5
        )

        if ik_solution is None:
            return False
        else:
            self.target_ee_pose = (target_position, target_orientation)
            self.target_joint_angles = list(ik_solution)
            if self.close:
                self.target_joint_angles[-1] = 0.0
                self.target_joint_angles[-2] = 0.0
            else:
                self.target_joint_angles[-1] = 0.8
                self.target_joint_angles[-2] = 0.8
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
    

    def get_joint_angles(self) -> List[float]:
        """
        Get the current joint angles for all controlled joints.

        Returns:
            List of joint angles in radians.
        """
        joint_angles = []
        for joint_index in self.controlled_joints:
            joint_state = p.getJointState(self.robot_id, joint_index)
            joint_angle = joint_state[0]  # position
            joint_angles.append(joint_angle)
        
        self.joint_angles = joint_angles
        return joint_angles


    def is_target_reached(self, position_tolerance: float = 0.02, orientation_tolerance: float = 0.5) -> bool:
        """
        Check if current pose is close enough to target pose.
        
        Args:
            position_tolerance: Maximum allowed position error in meters
            orientation_tolerance: Maximum allowed orientation error in radians
            
        Returns:
            bool: True if target is reached, False otherwise
        """
        self.get_ee_pose()
            
        # Calculate position error
        pos_error = np.linalg.norm(np.array(self.target_ee_pose[0]) - np.array(self.ee_pose[0]))
        
        # Calculate orientation error
        target_rot = R.from_quat(self.target_ee_pose[1])
        current_rot = R.from_quat(self.ee_pose[1])
        rot_error = (target_rot.inv() * current_rot).magnitude()
        
        return pos_error < position_tolerance and rot_error < orientation_tolerance

    
    def is_gripper_closed(self) -> bool:
        finger_1 = self.get_joint_angles()[-1]
        finger_2 = self.get_joint_angles()[-2]
        return np.abs(finger_1) < 0.001 and np.abs(finger_2) < 0.001


    def close_gripper(self) -> None:
        self.close = True


    def open_gripper(self) -> None:
        self.close = False  