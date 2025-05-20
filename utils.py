import os
import imageio.v2 as imageio
import numpy as np
import re
import ast
from typing import Any, List, Tuple

import scipy.spatial.transform as tf

def save_rgbd_image(color_img: np.ndarray, depth_img: np.ndarray, save_dir: str = "images", prefix: str = "frame", frame_idx: int = 0) -> None:
    """
    Save color and depth images to the specified directory.

    Args:
        color_img (np.ndarray): HxWx3 color image (dtype=uint8).
        depth_img (np.ndarray): HxW depth image (float32 or uint16).
        save_dir (str): Directory where images will be saved.
        prefix (str): File name prefix.
        frame_idx (int): Frame index to include in file name.
    """
    os.makedirs(save_dir, exist_ok=True)

    color_path = os.path.join(save_dir, f"color_{frame_idx:04d}.png")
    depth_path = os.path.join(save_dir, f"depth_{frame_idx:04d}.png")

    imageio.imwrite(color_path, color_img)

    # Convert float32 depth to 16-bit if needed (scaling to millimeters)
    if depth_img.dtype == np.float32 or depth_img.dtype == np.float64:
        scaled_depth = (depth_img * 1000).astype(np.uint16)  # meters to millimeters
    else:
        scaled_depth = depth_img

    imageio.imwrite(depth_path, scaled_depth)

    print(f"Saved color image to {color_path}")
    print(f"Saved depth image to {depth_path}")


def parse_answer(output: str) -> Any:
    """
    Parse the last Python code block in the output and extract the value of the variable `answer`.
    Returns the evaluated value if successful, otherwise returns "error".
    """
    try:
        # Find all Python code blocks: ```python ... ``` or '''python ... '''
        blocks = re.findall(
            r"(?:```|''')python\s*(.*?)\s*(?:```|''')",
            output,
            flags=re.DOTALL | re.IGNORECASE
        )

        if not blocks:
            return "error"

        # Use the last matched code block
        code = blocks[-1]

        # Parse with AST and look for: answer = <expr>
        tree = ast.parse(code, mode="exec")
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "answer":
                        return ast.literal_eval(node.value)

        return "error"

    except Exception:
        return "error"


def quaternion_to_R(quaternion: List[float]) -> np.ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix.
    """
    return tf.Rotation.from_quat(quaternion).as_matrix()


def quaternion_to_euler_degrees(quaternion: List[float]) -> List[float]:
    """
    Convert a quaternion to a roll-pitch-yaw Euler angles in degrees.
    """
    return tf.Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)


def R_to_euler_degrees(R: np.ndarray) -> List[float]:
    """
    Convert a 3x3 rotation matrix to a roll-pitch-yaw Euler angles.
    """
    return tf.Rotation.from_matrix(R).as_euler('xyz', degrees=True)


def R_to_quaternion(R: np.ndarray) -> List[float]:
    """
    Convert a 3x3 rotation matrix to a quaternion.
    """
    return tf.Rotation.from_matrix(R).as_quat()


def R_and_p_to_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix and a position vector to a 4x4 transformation matrix.
    
    Args:
        R: 3x3 rotation matrix
        p: 3x1 position vector
        
    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def T_to_R_and_p(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 4x4 transformation matrix to a rotation matrix and a position vector.
    """
    return T[:3, :3], T[:3, 3]


def transform_body_to_world(point: np.ndarray, body_position: List[float], body_orientation: List[float]) -> np.ndarray:
    """
    Transform a point from body frame to world frame.
    
    Args:
        point: Point in body frame [x, y, z]
        body_position: Position of body frame origin in world frame [x, y, z]
        body_orientation: Orientation of body frame in world frame as quaternion [x, y, z, w]
        
    Returns:
        Point in world frame [x, y, z]
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_R(body_orientation)
    
    # Transform point
    return np.array(body_position) + R @ np.array(point)


def round_for_print(variables, decimal_places: int = 2) -> List[float]:
    """
    Round a list of variables to 3 decimal places.
    Variable is a list of float numbers or a numpy array.
    """
    if isinstance(variables, list):
        return [round(var, decimal_places) for var in variables]
    elif isinstance(variables, np.ndarray):
        return list(np.round(variables, decimal_places))
    else:
        try:
            return list(np.round(np.array(variables), decimal_places))
        except:
            return variables
