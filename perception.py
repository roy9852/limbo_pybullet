from PIL import Image
import imageio.v3 as iio
import numpy as np
from lang_sam import LangSAM
import open3d as o3d
from sklearn.cluster import KMeans
import os

from config import config
from utils import *


def get_grasp_candidates(object_class: str, frame_id: str, save_path: str = "images", bounding_box: bool = False):
    # Load image
    color_path = f"{save_path}/color_{frame_id}.png"
    depth_path = f"{save_path}/depth_{frame_id}.png"

    color_img = iio.imread(color_path)  
    depth_img = iio.imread(depth_path)

    # Convert to np.array
    color_img = np.array(color_img)
    depth_img = np.array(depth_img)

    # Convert depth from millimeters to meters
    depth_img = depth_img.astype(np.float32) / 1000.0

    # Open-vocabulary segmentation
    lang_seg_model = LangSAM()

    image_pil = Image.fromarray(color_img)
    image_pil = image_pil.convert("RGB")

    results = lang_seg_model.predict([image_pil], [object_class])
    result = results[0]

    mask = np.array(result['masks'][0], dtype=bool)

    # Get the camera intrinsic parameters
    fov_rad = np.deg2rad(config.camera.fov)
    fy = config.camera.image_height / (2 * np.tan(fov_rad / 2))
    fx = fy * config.camera.aspect
    cx = config.camera.image_width / 2
    cy = config.camera.image_height / 2

    K = np.array([[fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]])
    
    # Get pixel grid
    H, W = depth_img.shape
    i, j = np.indices((H, W))
    z = depth_img
    x = (j - cx) * z / fx
    y = (i - cy) * z / fy
    points = np.stack((x, y, z), axis=-1)

    # Apply mask
    valid_mask = mask & (z > 0)
    points = points[valid_mask]                           
    colors = color_img[valid_mask] / 255.0    

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Filter outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Get object's oriented bounding box
    object_obb = pcd.get_oriented_bounding_box()
    object_obb.color = (1, 0, 0)

    # pcd visualization
    # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # Adjust size as needed
    # o3d.visualization.draw_geometries([pcd, coord_frame])
    # return

    # Get partial point cloud
    num_groups = 5
    kmeans = KMeans(n_clusters=num_groups, random_state=42)

    labels = kmeans.fit_predict(points)
    
    partial_pcds = []
    partial_obbs = []
    for i in range(num_groups):
        mask = labels == i
        group = o3d.geometry.PointCloud()
        group.points = o3d.utility.Vector3dVector(points[mask])
        group.colors = o3d.utility.Vector3dVector(colors[mask])
        partial_pcds.append(group)
        partial_obbs.append(group.get_oriented_bounding_box())
        partial_obbs[-1].color = (0.5, 0, 0.5) # purple
    
    # Visualize
    viewpoint = [0, 0, -1]
    up = [0, -1, 0]
    zoom = 0.5
    lookat = object_obb.center

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    ctr = vis.get_view_control()

    for i in range(0, num_groups+1):
        # Clear the scene
        vis.clear_geometries()

        # Get partial OBB
        if i == 0:
            grasp_obb = object_obb
        else:
            # Get the grasp OBB
            grasp_obb = partial_obbs[i-1]

        # If bounding box is True, visualize the OBB
        if bounding_box:
            # Create coordinate frame at the center of the OBB
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            coord_frame.translate(grasp_obb.center)

            # Get the rotation matrix from the OBB
            R = grasp_obb.R
            coord_frame.rotate(R, center=grasp_obb.center)

            # Create a filled box mesh
            box_mesh = o3d.geometry.TriangleMesh.create_box(
                width=grasp_obb.extent[0],
                height=grasp_obb.extent[1],
                depth=grasp_obb.extent[2]
            )
            box_mesh.translate(-grasp_obb.extent / 2)  # Center the box
            box_mesh.rotate(R, center=(0, 0, 0))  # Rotate to match OBB orientation
            box_mesh.translate(grasp_obb.center)  # Move to OBB center
            box_mesh.paint_uniform_color((0.5, 0, 0.5))  # Purple color

            geometries = [pcd, box_mesh, coord_frame]
        else:
            if i == 0:
                geometries = [pcd]
            else:
                geometries = [pcd, partial_pcds[i-1].paint_uniform_color((1, 0, 0))]

        for geom in geometries:
            vis.add_geometry(geom)
        
        # Set camera just once
        if viewpoint is not None:
            ctr.set_front(viewpoint)
        if up is not None:
            ctr.set_up(up)
        if zoom is not None:
            ctr.set_zoom(zoom)
        if lookat is not None:
            ctr.set_lookat(lookat)
        
        # Capture the image
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(save_path, f"grasp_candidate_{i}.png"))

    vis.destroy_window()
    
    print("Center, Extent, and Volume of OBBs:")
    print(f"{np.round(object_obb.center * 100, 1)} cm / {np.round(object_obb.extent * 100, 1)} cm / {np.round(np.prod(object_obb.extent)* 10**6, 1)} ml")
    for i in range(num_groups):
        print(f"{np.round(partial_obbs[i].center * 100, 1)} cm / {np.round(partial_obbs[i].extent * 100, 1)} cm / {np.round(np.prod(partial_obbs[i].extent)* 10**6, 1)} ml")

    return object_obb, partial_obbs

# Example usage with default camera parameters
object_obb, partial_obbs = get_grasp_candidates(object_class='hammer', frame_id="0001", bounding_box = True)

hand_position = [-0.008, 0.001, 0.306]
hand_orientation = [0.0, 1.0, 0.001, 0.001]
camera_position = [-0.784, -0.000, 0.567]
camera_orientation = [0.575, 0.575, 0.411, 0.412]

def obb_in_world_frame(obb: o3d.geometry.OrientedBoundingBox):
    R_pcd_to_object = obb.R
    p_pcd_to_object = obb.center
    T_pcd_to_object = R_and_p_to_T(R_pcd_to_object, p_pcd_to_object)

    R_world_to_camera = quaternion_to_R(camera_orientation)
    p_world_to_camera = camera_position
    T_world_to_camera = R_and_p_to_T(R_world_to_camera, p_world_to_camera)

    R_camera_to_pcd = np.array([[-1.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0],
                                [0.0, 0.0, 1.0]])
    p_camera_to_pcd = np.array([0.0, 0.0, 0.0])
    T_camera_to_pcd = R_and_p_to_T(R_camera_to_pcd, p_camera_to_pcd)

    T_world_to_object = T_world_to_camera @ T_camera_to_pcd @ T_pcd_to_object
    R_world_to_object, p_world_to_object = T_to_R_and_p(T_world_to_object)
    euler_world_to_object = R_to_euler_degrees(R_world_to_object)

    print(f"R_world_to_object: {R_world_to_object}")
    print(f"euler_world_to_object: {round_for_print(euler_world_to_object)}")
    print(f"p_world_to_object: {round_for_print(p_world_to_object)}")

    return R_world_to_object, euler_world_to_object, p_world_to_object


R_world_to_object, euler_world_to_object, p_world_to_object = obb_in_world_frame(partial_obbs[4])

target_vector = -R_world_to_object[:, 2]
target_vector = target_vector / np.linalg.norm(target_vector)
target_position = p_world_to_object - target_vector * 0.1

print(f"target_position: {round_for_print(target_position, 4)}")
print(f"target_vector: {round_for_print(target_vector, 4)}")