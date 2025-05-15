import itertools
import pickle
from functools import partial
from pathlib import Path

import rootutils
import torch

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from collections import defaultdict

import numpy as np
from smplx import MANOLayer

from src.utils.camera import load_cam_infos, project_to_2d
from src.utils.easy_convert import convert_rot_rep

PATH_TO_RECORDING = Path("data", "20250227_Testing")
PATH_TO_OUTPUT = PATH_TO_RECORDING / "Marshall" / "predictions" / "hands_processed"
PATH_TO_OUTPUT.mkdir(exist_ok=True)

CAM_INFOS = load_cam_infos(PATH_TO_RECORDING)


def mano_params_to_2d(mano_layer, global_orientation_mat, translation, K, T_camera_world):
    mano_output = mano_layer(
        global_orient=global_orientation_mat,  # (1, 1, 3, 3)
        transl=translation,  # (1, 3)
    )

    kpts = mano_output.vertices[0].float()

    points_2d = project_to_2d(kpts.T, K, T_camera_world)

    return points_2d


def compute_adjusted_mano_params(global_orientation_mat, translation, wrist_position, T_camera_world):
    # [INFO] We need this, since we rotate originally around the coordinate origin, Global orient rotates locally
    global_orientation_mat = T_camera_world[:3, :3].T @ global_orientation_mat
    translation_offset = (T_camera_world[:3, :3].T @ wrist_position) - wrist_position
    # Convert to (1, 3)
    translation_offset = translation_offset.T
    translation = (T_camera_world[:3, :3].T @ (-T_camera_world[:3, 3:4] + translation.T)).T + translation_offset

    return global_orientation_mat, translation


def get_camera_rays(points_3d, Ks, T_camera_worlds):
    """
    Calculate camera positions and ray directions

    Args:
        points_3d: List of 3D points in world coordinates (N x 3)
        Ks: List of camera intrinsic matrices (N x 3 x 3)
        T_camera_worlds: List of camera-to-world transformation matrices (N x 4 x 4)

    Returns:
        camera_positions: Camera centers in world space (N x 3)
        ray_directions: Ray directions in world space (N x 3)
    """
    n_cameras = len(points_3d)
    camera_positions = []
    ray_directions = []

    for i in range(n_cameras):
        # Get camera parameters
        K = Ks[i]
        T_camera_world = T_camera_worlds[i]

        # Camera position is the translation part of T_camera_world
        camera_pos = T_camera_world[:3, 3]
        camera_positions.append(camera_pos)

        # Calculate ray direction in camera coordinates
        # (assuming the point_3d is already in world space)
        point_world = points_3d[i]

        # Ray direction is from camera center to the point
        ray_dir = point_world - camera_pos
        ray_dir = ray_dir / torch.norm(ray_dir)
        ray_directions.append(ray_dir)

    return torch.stack(camera_positions), torch.stack(ray_directions)


def triangulate_rays(camera_positions, ray_directions):
    """
    Find the 3D point closest to the intersection of multiple rays (updated for newer PyTorch)

    Args:
        camera_positions: Camera centers in world space (N x 3)
        ray_directions: Normalized ray directions in world space (N x 3)

    Returns:
        Point in 3D space that minimizes the sum of squared distances to all rays
    """
    n_cameras = len(camera_positions)

    # Convert inputs to tensors if they aren't already
    camera_positions = torch.as_tensor(camera_positions, dtype=torch.float32)
    ray_directions = torch.as_tensor(ray_directions, dtype=torch.float32)

    # Normalize ray directions
    ray_directions = ray_directions / torch.norm(ray_directions, dim=1, keepdim=True)

    # Set up the linear system to find the closest point to all rays
    A = torch.zeros((n_cameras * 3, 3), dtype=torch.float32)
    b = torch.zeros(n_cameras * 3, dtype=torch.float32)

    for i in range(n_cameras):
        camera_pos = camera_positions[i]
        ray_dir = ray_directions[i]

        ray_outer = torch.outer(ray_dir, ray_dir)
        block = torch.eye(3) - ray_outer

        A[i * 3 : (i + 1) * 3, :] = block
        b[i * 3 : (i + 1) * 3] = torch.matmul(block, camera_pos)

    # Solve the least squares problem
    solution = torch.linalg.lstsq(A, b.unsqueeze(1)).solution
    optimal_point = solution[:3, 0]

    return optimal_point


def triangulate_from_cameras(points_3d, T_camera_worlds):
    """
    Triangulate 3D point from multiple world space points using ray intersection

    Args:
        points_3d: List of 3D points in world coordinates (N x 3)
        T_camera_worlds: List of world-to-camera transformation matrices (N x 4 x 4)
                        (defined as p_camera = T_camera_worlds @ p_world)

    Returns:
        Optimal 3D point in world coordinates
    """
    n_cameras = len(points_3d)
    camera_positions = []
    ray_directions = []

    for i in range(n_cameras):
        # Extract camera center in world coordinates
        T_world_camera = torch.inverse(T_camera_worlds[i])
        origin_camera = torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        camera_pos_homogeneous = torch.matmul(T_world_camera, origin_camera)
        camera_pos = camera_pos_homogeneous[:3] / camera_pos_homogeneous[3]
        camera_positions.append(camera_pos)

        # The ray direction is from camera center to the world point
        point_world = points_3d[i]
        ray_dir = point_world - camera_pos
        ray_dir = ray_dir / torch.norm(ray_dir)
        ray_directions.append(ray_dir)

    # Convert to tensors
    camera_positions = torch.stack(camera_positions)
    ray_directions = torch.stack(ray_directions)

    # Find closest point to all rays
    optimal_point = triangulate_rays(camera_positions, ray_directions)

    return optimal_point


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    mano_layer_fn = partial(MANOLayer, model_path="body_models/mano", create_body_pose=False)

    rh_mano_layer = mano_layer_fn(is_rhand=True).to(device)
    lh_mano_layer = mano_layer_fn(is_rhand=False).to(device)

    mano_layer_dict = {"right": rh_mano_layer, "left": lh_mano_layer}

    for layer in mano_layer_dict.values():
        layer.eval()
        layer.requires_grad_(False)

    pred_path = Path("data/20250227_Testing/Marshall/predictions/hands/obj")
    pred_files = sorted(pred_path.iterdir())

    for pred_file in pred_files:
        with pred_file.open("rb") as f:
            preds = pickle.load(f)

        frame_num = int(pred_file.stem.split("_")[-1])

        hand_predictions = defaultdict(list)
        for side, cam_num in itertools.product(["left", "right"], [1, 2]):
            cam_params = CAM_INFOS[f"camera_0{cam_num + 4}"]

            if cam_num not in preds:
                continue

            cam_preds = [
                x["wilor_preds"]
                for x in preds[cam_num]
                if (x["is_right"] and side == "right") or (not x["is_right"] and side == "left")
            ]
            if not cam_preds:
                continue
            else:
                cam_preds = cam_preds[0]

            # (1, 3)
            global_orient_aa = torch.from_numpy(cam_preds["global_orient"]).float()
            # (1, 3, 3)
            global_orientation_mat = convert_rot_rep("aa -> rotation_matrix", global_orient_aa)
            # (1, 3)
            translation = torch.from_numpy(cam_preds["pred_cam_t_full"]).float()
            # (15, 3)
            hand_pose_aa = torch.from_numpy(cam_preds["hand_pose"]).float()
            # (1, 10)
            betas = torch.from_numpy(cam_preds["betas"]).float()

            K = torch.from_numpy(cam_params["K"]).float()
            T_camera_world = torch.from_numpy(np.linalg.inv(cam_params["T_world_camera"])).float()

            wrist_position = torch.from_numpy(cam_preds["pred_keypoints_3d"][0, 0:1].T)

            global_orientation_mat, translation = compute_adjusted_mano_params(
                global_orientation_mat, translation, wrist_position, T_camera_world
            )

            global_orient_aa = convert_rot_rep("rotation_matrix -> aa", global_orientation_mat)

            hand_predictions[side].append(
                {
                    "hand_pose_aa": hand_pose_aa,
                    "betas": betas,
                    "global_orient_aa": global_orient_aa,
                    "translation": translation,
                    "K": K,
                    "T_camera_world": T_camera_world,
                }
            )

        print("Consolidating Meshes now")

        data = []
        for side in ["left", "right"]:
            predictions = hand_predictions[side]
            if not predictions:
                data.append(np.zeros(109, dtype=np.float32))
                continue
            # Consolidating by taking the mean
            # (2, 1, 15, 3)
            hand_pose_aa = torch.stack([x["hand_pose_aa"] for x in predictions])
            # (2, 1, 10)
            betas = torch.stack([x["betas"] for x in predictions])
            # (2, 1, 1, 3)
            global_orient_aa = torch.stack([x["global_orient_aa"] for x in predictions])
            # (2, 1, 3)
            translation = torch.stack([x["translation"] for x in predictions])
            # Ks = torch.stack([x["K"] for x in predictions])
            T_camera_worlds = torch.stack([x["T_camera_world"] for x in predictions])

            translation = triangulate_from_cameras(translation.squeeze(1), T_camera_worlds)
            translation = translation.unsqueeze(0)

            # [INFO] Consolidating here
            hand_pose_aa = hand_pose_aa.mean(0)
            betas = betas.mean(0)
            global_orient_aa = global_orient_aa.mean(0)
            # translation = translation.mean(0)

            hand_pose_mat = convert_rot_rep("aa -> rotation_matrix", hand_pose_aa)
            global_orientation_mat = convert_rot_rep("aa -> rotation_matrix", global_orient_aa)

            partial_mano_layer = partial(
                mano_layer_dict[side], betas=betas.to(device), hand_pose=hand_pose_mat.to(device)
            )

            mano_output = partial_mano_layer(
                global_orient=global_orientation_mat.to(device),  # (1, 1, 3, 3)
                transl=translation.to(device),  # (1, 3)
            )

            # [10 + 45 + 3 + 3 + 48]
            data.append(
                np.concat(
                    [
                        betas.reshape(-1).cpu().numpy(),
                        hand_pose_aa[0].reshape(-1).cpu().numpy(),
                        global_orient_aa.reshape(-1).cpu().numpy(),
                        translation.reshape(-1).cpu().numpy(),
                        mano_output.joints.reshape(-1).cpu().numpy(),
                    ]
                )
            )

        np.save(PATH_TO_OUTPUT / f"frame_{frame_num:06d}.npy", np.stack(data))

        # # VISUALIZATION
        # imgs = {}
        # for side, hand_params in output.items():
        #     hand_pose_aa = hand_params["hand_pose_aa"]
        #     betas = hand_params["betas"]
        #     global_orient_aa = hand_params["global_orient_aa"]
        #     translation = hand_params["translation"]

        #     hand_pose_mat = convert_rot_rep("aa -> rotation_matrix", hand_pose_aa)
        #     global_orientation_mat = convert_rot_rep("aa -> rotation_matrix", global_orient_aa)

        #     partial_mano_layer = partial(
        #         mano_layer_dict[side], betas=betas.to(device), hand_pose=hand_pose_mat.to(device)
        #     )

        #     for cam_num in [1, 2]:
        #         cam_params = CAM_INFOS[f"camera_0{cam_num + 4}"]
        #         if cam_num not in imgs:
        #             img = cv2.imread(f"./data/20250227_Testing/Marshall/recording/export/color_{frame_num:06d}_camera0{cam_num}.jpg")
        #             img = cv2.undistort(
        #                 img,
        #                 cam_params["K"],
        #                 np.array(
        #                     [cam_params["radial_params"][0]]
        #                     + [cam_params["radial_params"][1]]
        #                     + list(cam_params["tangential_params"][:2])
        #                     + [cam_params["radial_params"][2]]
        #                     + [0, 0, 0]
        #                 ),
        #             )
        #             imgs[cam_num] = img
        #         else:
        #             img = imgs[cam_num]

        #         H, W, C = img.shape

        #         K = cam_params["K"]
        #         K[0, -1] = W / 2
        #         K[1, -1] = H / 2

        #         T_camera_world = np.linalg.inv(cam_params["T_world_camera"])

        #         K = torch.from_numpy(K).float()
        #         T_camera_world = torch.from_numpy(T_camera_world).float()

        #         partial_mano_layer = partial(
        #             mano_layer_dict[side], betas=betas.to(device), hand_pose=hand_pose_mat.to(device)
        #         )

        #         points_2d = mano_params_to_2d(
        #             partial_mano_layer,
        #             global_orientation_mat.to(device),
        #             translation.to(device),
        #             K.to(device),
        #             T_camera_world.to(device),
        #         )

        #         points_2d = points_2d.detach().cpu().numpy()

        #         for x, y in points_2d.T:
        #             cv2.circle(img, (int(x), int(y)), 2, (255, 175, 0), -1)

        # cv2.imwrite(f"test_{frame_num}.jpg", np.hstack(list(imgs.values())))


if __name__ == "__main__":
    main()
