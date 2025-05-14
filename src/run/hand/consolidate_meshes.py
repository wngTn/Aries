import pickle
from pathlib import Path
from functools import partial

import cv2
import rootutils
import torch
import torch.optim as optim
from tqdm import tqdm  # Added tqdm import

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

import numpy as np
from smplx import MANOLayer

from src.utils.camera import load_cam_infos, project_to_2d
from src.utils.easy_convert import convert_rot_rep

PATH_TO_RECORDING = Path("data", "20250227_Testing")

CAM_INFOS = load_cam_infos(PATH_TO_RECORDING)


def mano_params_to_2d(mano_layer, global_orientation_aa, translation, K, T_camera_world):
    global_orientation_mat = convert_rot_rep("aa -> rotation_matrix", global_orientation_aa)

    mano_output = mano_layer(
        global_orient=global_orientation_mat,  # (1, 1, 3, 3)
        transl=translation,  # (1, 3)
    )

    kpts = mano_output.joints[0].float()

    points_2d = project_to_2d(kpts.T, K, T_camera_world)

    return points_2d


def error_fn(mano_layer, global_orientation_aa, translation, target_points_2d, K, T_camera_world):
    """
    Calculate the error between the target 2D points and the projected 2D points using the MANO layer.
    """
    # Convert global_orientation_aa to rotation matrix
    global_orientation_mat = convert_rot_rep("aa -> rotation_matrix", global_orientation_aa)

    # Run the MANO model to get the predicted 2D keypoints
    mano_output = mano_layer(global_orient=global_orientation_mat, transl=translation)

    # Project 3D keypoints to 2D
    kpts = mano_output.joints[0].float()
    points_2d = project_to_2d(kpts.T, K, T_camera_world)

    # Compute the L2 error between the target and predicted 2D points
    error = torch.norm(points_2d - target_points_2d)
    print(error)
    return error


def transform(mano_layer, target_points_2d, K, T_camera_world):
    """
    Optimize the global orientation and translation to minimize the error between the projected 2D points and target 2D points.
    """
    # Initialize global orientation and translation (starting point for optimization)
    global_orientation_aa = torch.zeros(
        (1, 1, 3), dtype=torch.float32, device=target_points_2d.device, requires_grad=True
    )
    translation = torch.zeros((1, 3), dtype=torch.float32, device=target_points_2d.device, requires_grad=True)

    # Use L-BFGS optimizer for optimization
    optimizer = optim.LBFGS([global_orientation_aa, translation], lr=1e-2, max_iter=4, line_search_fn="strong_wolfe")

    # Initialize variables for early stopping
    best_loss = float("inf")
    patience = 50
    counter = 0
    previous_loss = float("inf")

    def closure():
        optimizer.zero_grad()
        # Calculate the error (loss) and backward pass
        loss = error_fn(mano_layer, global_orientation_aa, translation, target_points_2d, K, T_camera_world)
        loss.backward()
        return loss

    for i in tqdm(range(100)):
        # Run the optimization
        current_loss = optimizer.step(closure)

        # Check for improvement
        if current_loss < best_loss:
            best_loss = current_loss
            counter = 0  # Reset counter when we see improvement
        # If no improvement or very minimal improvement
        elif (previous_loss - current_loss) < 1e-5:  # Small threshold for meaningful improvement
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {i + 1} iterations.")
                break

        previous_loss = current_loss

    return global_orientation_aa, translation


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    mano_layer = MANOLayer(
        model_path="body_models/mano",
        create_body_pose=False,
    ).to(device)

    mano_layer.requires_grad_(False)

    cam_num = 2

    img = cv2.imread(f"data/20250227_Testing/Marshall/recording/export/color_000400_camera0{cam_num}.jpg")

    cam_params = CAM_INFOS[f"camera_0{cam_num + 4}"]
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

    cam_preds = preds[cam_num][0]["wilor_preds"]

    # (1, 3)
    global_orient_aa = torch.from_numpy(cam_preds["global_orient"])
    # (1, 3)
    translation = torch.from_numpy(cam_preds["pred_cam_t_full"])
    # (15, 3)
    hand_pose_aa = torch.from_numpy(cam_preds["hand_pose"])
    # (1, 10)
    betas = torch.from_numpy(cam_preds["betas"])

    H, W, C = img.shape
    focal_length = cam_preds["scaled_focal_length"]
    K_pred = torch.tensor(
        [
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    T_camera_world = torch.eye(4, dtype=torch.float32)

    hand_pose_mat = convert_rot_rep("aa -> rotation_matrix", hand_pose_aa)

    partial_mano_layer = partial(mano_layer, betas=betas.to(device), hand_pose=hand_pose_mat.to(device))

    print("Calculating initial target points...")
    target_points_2d = mano_params_to_2d(
        partial_mano_layer,
        global_orient_aa.to(device),
        translation.to(device),
        K_pred.to(device),
        T_camera_world.to(device),
    )

    print("Starting optimization...")
    optim_global_orient_aa, optim_translation = transform(
        partial_mano_layer,
        target_points_2d.float().to(device),
        torch.from_numpy(cam_params["K"]).float().to(device),
        torch.from_numpy(np.linalg.inv(cam_params["T_world_camera"])).float().to(device),
    )

    points_2d = mano_params_to_2d(
        partial_mano_layer,
        optim_global_orient_aa.to(device),
        optim_translation.to(device),
        torch.from_numpy(cam_params["K"]).float().to(device),
        torch.from_numpy(np.linalg.inv(cam_params["T_world_camera"])).float().to(device),
    )

    points_2d = points_2d.detach().cpu().numpy()

    for x, y in points_2d.T:
        cv2.circle(
            img,
            (int(x), int(y)),
            2,
            (255, 175, 0),
            -1
        )

    cv2.imwrite("test.jpg", img)


if __name__ == "__main__":
    main()
