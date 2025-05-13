from pathlib import Path

import open3d as o3d
import rootutils
import numpy as np

rootutils.setup_root(__file__, ".project-root", pythonpath=True, dotenv=True)

from lib.utils.camera import load_cam_infos
from lib.utils.o3d_render import get_camera_poses, visualize_point_clouds

from scipy.spatial.transform import Rotation

r1 = Rotation.from_euler("x", 90, degrees=True)
r2 = Rotation.from_euler("y", 180, degrees=True)

# Combine the rotations
final_rotation = r2 * r1

# Get the rotation matrix
rotation_matrix = final_rotation.as_matrix()
rotation_matrix = np.eye(3)


def main():
    path_to_trial = Path("data/recordings/20250206_Testing")
    # Load camera information
    cam_infos = load_cam_infos(path_to_trial)
    # Load images and create point clouds
    color_image_path = path_to_trial / "color"
    depth_image_path = path_to_trial / "depth"
    # pcds = point_cloud_from_images(color_image_path, depth_image_path, cam_infos)

    # print(f"Loaded {len(pcds)} point clouds")

    geometry_dict = {}
    # for i, pcd in enumerate(pcds):
    #     geometry_dict[f"pcd_{i}"] = pcd
    geometry_dict["pcd"] = o3d.io.read_point_cloud(
        str(path_to_trial / "Orbbec" / "pointclouds_fused" / "pointcloud_000000.ply")
    )
    
    # geometry_dict["pcd"].points = o3d.utility.Vector3dVector(
    #     np.array(geometry_dict["pcd"].points) @ rotation_matrix.T
    # )

    aria_pcd = o3d.io.read_point_cloud(
        "test.ply"
    )

    geometry_dict["aria_pcd"] = aria_pcd
    aria2orbbec = np.array([
        [-0.4226183295249939, 0.9063077569007874, -7.92319454490098e-08, -0.14000000059604645],
        [0.9063077569007874, 0.4226183295249939, -3.694647077168156e-08, 0.25999999046325684],
        [0.0, -8.742277657347586e-08, -1.0, -0.4399999976158142],
        [0.0, 0.0, 0.0, 1.0]
    ])
    # Apply the 4x4 transformation matrix to the point cloud
    geometry_dict["aria_pcd"].transform(aria2orbbec)

    camera_poses = get_camera_poses(cam_infos, rotation=rotation_matrix)
    geometry_dict.update(camera_poses)

    visualize_point_clouds(geometry_dict)


if __name__ == "__main__":
    main()
