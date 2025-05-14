"""
Script for a simple consolidation of meshes
Data is in:
    - data
        - <recoording>
            - Marshall
                - predictions/hands/obj

.pkl file contains:
dict(
    cam_num = list({
       "hand_bbox": np.ndarray,
       "is_right": Literal[0.0, 1.0],
       "wilor_preds": {
           "global_orient" : np.ndarray (1, 1, 3),
           "hand_pose": np.ndarray (1, 15, 3),
           "betas": np.ndarray (1, 10),
           "pred_cam": np.ndarray (1, 3),
           "pred_keypoints_3d": np.ndarray (1, 21, 3),
           "pred_vertices": np.ndarray (1, 778, 3),
           "pred_cam_t_full": np.ndarray (1, 3),
           "scaled_focal_length": float,
       }
    }, ...)
    ...
)
"""

import pickle
from pathlib import Path
import torch
import cv2
import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils.camera import load_cam_infos, project_to_2d
from src.utils.easy_convert import convert_rot_rep
from smplx import MANOLayer

PATH_TO_RECORDING = Path("data", "20250227_Testing")

CAM_INFOS = load_cam_infos(PATH_TO_RECORDING)

import numpy as np

def draw_vertices_on_image_with_utils(image, vertices, cam_t, focal_length, K, T_camera_world):
    H, W, C = image.shape
    # vertices_3d: (3, N)
    vertices_3d = vertices[0].T

    K_pred = np.array(
        [
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1],
        ]
    )

    # Construct the extrinsic matrix
    T_camera_world_old = np.eye(4)
    # Set the translation part
    T_camera_world_old[:3, 3] = cam_t
    # Project using old parameters; old_points_2d: (2, N)
    old_points_2d = project_to_2d(vertices_3d, K_pred, T_camera_world_old)

    # ---- Transform vertices_3d to vertices_new ----
    # Step 1: Convert vertices_3d to homogeneous coordinates
    n_points = vertices_3d.shape[-1]
    p_world_hom_old = np.r_[vertices_3d, np.ones((1, n_points))]

    # Step 2: Transform vertices from old world to old camera space
    p_cam_old = T_camera_world_old @ p_world_hom_old

    # Step 3: Apply old intrinsics to get to normalized device coordinates (without dividing by z)
    p_img_hom_old = K_pred @ p_cam_old[:3]

    # Step 4: Get depths (z values) from old camera space
    z_old = p_cam_old[2]

    # Step 5: Create 3D points in new camera space that would project to the same 2D points
    # First, get the normalized device coordinates (divide by z)
    p_norm_img_old = p_img_hom_old[:2] / p_img_hom_old[2:3]

    # Step 6: Unproject to new camera space
    # Start with normalized coordinates [x/z, y/z, 1]
    p_norm_img_hom_old = np.r_[p_norm_img_old, np.ones((1, n_points))]

    # Apply inverse of new intrinsics to get directions in new camera space
    directions_cam = np.linalg.inv(K) @ p_norm_img_hom_old

    # Scale directions by the old depths to get 3D points in new camera space
    p_cam = directions_cam * z_old

    # Step 7: Transform from new camera space to new world space
    # First, add homogeneous coordinate
    p_cam_hom = np.r_[p_cam, np.ones((1, n_points))]

    # Apply inverse of new extrinsics to get to new world space
    p_world_hom = np.linalg.inv(T_camera_world) @ p_cam_hom

    # Extract the 3D coordinates
    p_world = p_world_hom[:3]

    new_points_2d = project_to_2d(p_world, K, T_camera_world)
    print(f"This should be 0: {np.linalg.norm(old_points_2d - new_points_2d)}")

    # Draw points on image
    for point in new_points_2d.T:
        cv2.circle(
            image,
            (int(point[0]), int(point[1])),
            3,
            (0, 0, 255),
            thickness=-1,
        )
    return image


def main():
    mano_layer = MANOLayer(
        model_path="body_models/mano",
        create_body_pose=False,
    )

    img = cv2.imread(
        "/home/tonyw/Aries/Aries/data/20250227_Testing/Marshall/recording/export/color_000400_camera02.jpg"
    )
    cam_params = CAM_INFOS["camera_05"]
    img = cv2.undistort(
        img,
        cam_params["K"],
        np.array(
            [cam_params["radial_params"][0]]
            + [cam_params["radial_params"][1]]
            + list(cam_params["tangential_params"][:2])
            + [cam_params["radial_params"][2]]
            + [0, 0, 0]
        ),
    )

    with open("data/20250227_Testing/Marshall/predictions/hands/obj/frame_000400.pkl", "rb") as f:
        preds = pickle.load(f)

    cam_01_preds = preds[2]
    
    mano_output = mano_layer(
        betas=torch.from_numpy(cam_01_preds[0]["wilor_preds"]["betas"]),
        global_orient=convert_rot_rep(
            "aa -> rotation_matrix", torch.from_numpy(cam_01_preds[0]["wilor_preds"]["global_orient"])
        ),
        hand_pose=convert_rot_rep(
            "aa -> rotation_matrix", torch.from_numpy(cam_01_preds[0]["wilor_preds"]["hand_pose"])
        )
    )

    img = draw_vertices_on_image_with_utils(
        img,
        cam_01_preds[0]['wilor_preds']["pred_vertices"],
        cam_01_preds[0]['wilor_preds']['pred_cam_t_full'],
        cam_01_preds[0]['wilor_preds']['scaled_focal_length'],
        cam_params["K"],
        np.linalg.inv(cam_params["T_world_camera"])
    )
    cv2.imwrite("test.jpg", img)


if __name__ == "__main__":
    main()
