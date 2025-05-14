import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import rootutils
import torch

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from external.wilor.wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from src.utils.camera import load_cam_infos
from src.utils.visualization.rendering import Renderer
from src.utils.camera import load_cam_infos, project_to_2d_np

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)
PATH_TO_IMAGES = Path("data", "20250227_Testing", "Marshall", "recording", "export")
OUTPUT_PATH_IMAGES = Path("data", "20250227_Testing", "Marshall", "predictions", "hands", "images")
OUTPUT_PATH_OBJ = Path("data", "20250227_Testing", "Marshall", "predictions", "hands", "obj")
FRAME_INTERVAL = [400, 14_000]
CAM_IDS = [1, 2]

CAM_INFOS = load_cam_infos(Path("data/20250227_Testing"))

def to_name(frame_idx, cam_idx):
    return f"color_{frame_idx:06d}_camera{cam_idx:02d}.jpg"


def load_frame_images(path_to_images: Path, frame_num):
    """Load images for a specific frame and return as a list."""
    frame_images = []
    for cam_num in CAM_IDS:
        img_name = to_name(frame_num, cam_num)
        img_path = path_to_images / img_name
        cam_params = CAM_INFOS[f"camera_0{cam_num + 4}"]
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            frame_images.append((cam_num, img))
        else:
            print(f"Warning: Image {img_path} not found")
    return frame_images


def frame_generator(path_to_images: Path):
    """Generator that yields (frame_num, frame_images) pairs for each frame."""
    for frame_num in range(FRAME_INTERVAL[0], FRAME_INTERVAL[1]):
        frame_images = load_frame_images(path_to_images, frame_num)
        if frame_images:  # Only yield frames that have images
            yield frame_num, frame_images


def predict_images(model, images):
    """Run hand prediction on a list of images."""
    outputs = []
    for cam_id, image in images:
        cam_params = CAM_INFOS[f"camera_0{cam_id + 4}"]
        t0 = time.time()
        prediction = model.predict(image, K=cam_params["K"])
        print(f"Prediction time: {time.time() - t0:.4f}s")
        outputs.append((cam_id, prediction))
    return outputs

def render_hands(renderer, frame_num, image, predictions, cam_id, output_path_images, output_path_obj):
    """Render hand predictions and save images and 3D models."""
    # Create output directories if they don't exist
    output_path_images.mkdir(parents=True, exist_ok=True)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Prepare image for rendering
    render_image = image.copy()
    render_image = render_image.astype(np.float32)[:, :, ::-1] / 255.0

    pred_keypoints_2d_all = []

    for i, out in enumerate(predictions):
        verts = out["wilor_preds"]["pred_vertices"][0]
        is_right = out["is_right"]
        cam_t = out["wilor_preds"]["pred_cam_t_full"][0]
        scaled_focal_length = out["wilor_preds"]["scaled_focal_length"]
        pred_keypoints_2d = out["wilor_preds"]["pred_keypoints_2d"]
        pred_keypoints_2d_all.append(pred_keypoints_2d)

        # Rendering arguments
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )

        # Render hand
        cam_view = renderer.render_rgba(
            verts, cam_t=cam_t, render_res=[image.shape[1], image.shape[0]], is_right=is_right, **misc_args
        )

        # Overlay image
        render_image = render_image[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]

    # Convert to uint8 for saving
    render_image = (255 * render_image).astype(np.uint8)

    # Draw keypoints
    for pred_keypoints_2d in pred_keypoints_2d_all:
        for j in range(pred_keypoints_2d[0].shape[0]):
            color = (0, 0, 255)  # Red in BGR
            radius = 3
            x, y = pred_keypoints_2d[0][j]
            cv2.circle(render_image, (int(x), int(y)), radius, color, -1)

    return render_image



def main():
    # Set up device and model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize model
    model = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float32, verbose=False)
    renderer = Renderer(model.wilor_model.mano.faces)

    # Create output directories
    OUTPUT_PATH_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH_OBJ.mkdir(parents=True, exist_ok=True)

    print(f"Processing images from {PATH_TO_IMAGES}")

    # Process each frame lazily using the generator
    for frame_num, images in frame_generator(PATH_TO_IMAGES):
        print(f"\nProcessing frame {frame_num}")

        # Predict hands for each camera view
        model_outputs = predict_images(model, images)

        rendered_images = []
        preds = {}
        # For visualization, render hands on each camera view separately
        for cam_id, img in images:
            # Filter predictions for this camera
            cam_predictions = [pred for cam_id_pred, preds in model_outputs if cam_id_pred == cam_id for pred in preds]

            if cam_predictions:
                # Render
                # rendered_image = render_hands(
                #     renderer, frame_num, img, cam_predictions, cam_id, OUTPUT_PATH_IMAGES, OUTPUT_PATH_OBJ
                # )
                # rendered_images.append(rendered_image)

                # K = CAM_INFOS[f"camera_0{cam_id + 4}"]["K"]
                # K[0, -1] = img.shape[1] / 2
                # K[1, -1] = img.shape[0] / 2
                # T_camera_world = np.linalg.inv(CAM_INFOS[f"camera_0{cam_id + 4}"]["T_world_camera"])
                
                # for cam_pred in cam_predictions:
                #     points_3d = cam_pred["wilor_preds"]["pred_vertices"][0].T
                #     # T_camera_world[:3, :3] = np.eye(3)
                #     points_3d = points_3d - T_camera_world[:3, 3:4] + cam_pred["wilor_preds"]["pred_cam_t_full"].T
                #     points_3d = T_camera_world[:3, :3].T @ points_3d
                #     # T_camera_world[:3, 3] = cam_pred["wilor_preds"]["pred_cam_t_full"][0]
                #     points_2d = project_to_2d_np(points_3d, K, T_camera_world)
                #     for x, y in points_2d.T:
                #         cv2.circle(
                #             img,
                #             (int(x), int(y)),
                #             2,
                #             (255, 175, 0),
                #             -1
                #         )
                # rendered_images.append(img)
                
                preds[cam_id] = cam_predictions

        if rendered_images:
            img_filename = f"frame_{frame_num:06d}.jpg"
            combined_image = np.hstack(rendered_images)
            combined_image = cv2.resize(combined_image, np.array(combined_image.shape[:2])[::-1] // 4)
            cv2.imwrite(str(OUTPUT_PATH_IMAGES / img_filename), combined_image)

        if preds:
            pkl_filename = f"frame_{frame_num:06d}.pkl"
            with open(OUTPUT_PATH_OBJ / pkl_filename, "wb") as f:
                pickle.dump(preds, f)
            print(f"Saved visualization for frame {frame_num}")

    print("\nProcessing complete!")
    print(f"Image visualizations saved to: {OUTPUT_PATH_IMAGES}")
    print(f"3D model files saved to: {OUTPUT_PATH_OBJ}")


if __name__ == "__main__":
    main()
