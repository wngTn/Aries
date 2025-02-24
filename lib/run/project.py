from pathlib import Path

import cv2
import numpy as np
import rootutils
from copy import deepcopy
import trimesh

rootutils.setup_root(__file__, ".project-root", pythonpath=True, dotenv=True)

from scipy.spatial.transform import Rotation

from lib.utils.camera import load_cam_infos, project_to_2d
from lib.utils.image import undistort_image

COLORS = [
    (255, 0, 0), # Red
    (0, 255, 0), # Green
    (0, 0, 255), # Blue
    (128, 0, 128), # Purple 
]

# Rotation matrix for marshall cameras. Rotate -90 degrees around x-axis
T_orbbec_marshall = Rotation.from_euler("x", -90, degrees=True).as_matrix()

mesh = trimesh.load("simulation_board.stl")

# mesh_transformation = np.array([
# [1.0, 0.0, 0.0, -0.05888441950082779],
#         [0.0, -4.371138828673793e-08, -1.0, 0.19682589173316956],
#         [0.0, 1.0, -4.371138828673793e-08, -0.06605557352304459],
#         [0.0, 0.0, 0.0, 1.0]
# ])

# mesh.apply_transform(mesh_transformation)

hand_points = np.load("/Users/tonywang/Documents/hiwi/Aries/Code/Aries/data/recordings/20250206_Testing/Aria/export/hand/hand_000000.npy", allow_pickle=True).item()

points_3d_aria_world = np.array([v for k, v in hand_points.items() if not "normal" in k])
points_3d_aria_world = np.concatenate([points_3d_aria_world, np.ones((points_3d_aria_world.shape[0], 1))], axis=1).T

T_orbbec_aria = np.array(
    [[-0.8748278021812439, -0.4844340682029724, 1.0859806565122199e-07, -0.26741909980773926],
        [7.647987843029114e-08, 8.606196644223019e-08, 1.0, 0.4549257457256317],
        [-0.4844340682029724, 0.8748278021812439, -3.823993921514557e-08, -0.14493417739868164],
        [0.0, 0.0, 0.0, 1.0]])

points_orbbec = T_orbbec_aria @ points_3d_aria_world
points_3d = points_orbbec[:3].T

# swap x and z axis
aria_rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
points_3d = points_3d @ aria_rotation.T

pcd = trimesh.sample.sample_surface(mesh, 1000)[0]

def main():
    path_to_trial = Path("data/recordings/20250206_Testing")

    cameras = ["camera01", "camera02", "camera03", "camera04", "camera05", "camera06"]
    images = []

    # Load camera information
    cam_infos = load_cam_infos(path_to_trial)
    
    marshall_path = Path(f"{path_to_trial}/Marshall/recording/export")
    orrbec_path = Path(f"{path_to_trial}/Orbbec/color")
    

    # points_3d = np.array([
    #     [0.54, 1, 0.54],
    #     [-0.54, 1, 0.54],
    #     [-0.54, 1, -0.54],
    #     [0.54, 1, -0.54],
    # ]) / 2


    # points_3d = np.array(pcd)
    # points_3d = np.array([
    #     [0, 0, 0]
    # ])

    for cam in cameras:
        cam_params = cam_infos[cam]
        cam_id = int(cam[-1])
        if cam_id >= 5:
            img = cv2.imread(
                str(marshall_path / f"color_000000_camera{cam_id-4:02d}.jpg")
            )
            img = cv2.undistort(img, cam_params['intrinsics'], np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0]))
            # [INFO] points_3d.shape = (N, 3)
            _points_3d = (T_orbbec_marshall @ deepcopy(points_3d).T).T
            T_marshall_orbbec = np.linalg.inv(T_orbbec_marshall)
        else:
            img = cv2.imread(
                str(orrbec_path / f"color_000000_camera{cam_id:02d}.jpg")
            )
            img = undistort_image(img, cam_params, 'color')
            _points_3d = deepcopy(points_3d)
            
        # List to store the 2D points on the image
        points_2d = []

        for i, point_3d in enumerate(_points_3d):
            # Project 3D point to 2D image coordinates
            point_2d = project_to_2d(
                point_3d, cam_params["intrinsics"], np.linalg.inv(cam_params["extrinsics"]) if cam_id < 5 else cam_params["extrinsics"]
            )

            # Save the 2D point
            points_2d.append((int(point_2d[0]), int(point_2d[1])))

            # Draw the point on the image
            cv2.circle(
                img,
                (int(point_2d[0]), int(point_2d[1])),
                10,
                COLORS[i % len(COLORS)],
                thickness=-1,
            )
        
        # Draw lines between the points (if there are multiple points)
        # for i in range(len(points_2d)):
        #     for j in range(i + 1, len(points_2d)):
        #         cv2.line(img, points_2d[i], points_2d[j], (255, 0, 0), 2)

        images.append(img)
        
    # Determine the number of rows needed
    num_columns = 2
    num_rows = (len(images) + num_columns - 1) // num_columns

    # Create a blank canvas to place the images
    height, width, _ = images[0].shape
    canvas = np.zeros((height * num_rows, width * num_columns, 3), dtype=np.uint8)

    # Place each image in the canvas
    for idx, img in enumerate(images):
        row = idx // num_columns
        col = idx % num_columns
        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = img

    # cv2.imshow("image", canvas)
    # cv2.waitKey(0)
    cv2.imwrite("test.jpg", canvas)


if __name__ == "__main__":
    main()
