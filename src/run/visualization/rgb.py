from pathlib import Path

import cv2
import pyarrow.dataset as ds
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, ".project-root", pythonpath=True, dotenv=True)

import numpy as np
from scipy.spatial.transform import Rotation

from lib.utils.camera import load_cam_infos, project_to_2d
from lib.utils.image import create_collage, create_video_from_images, undistort_image


def process_video(
    cam_infos,
    orbbe_timestamps,
    input_dirs,
    output_path,
    skip_n_frames=2,
    fps=6,
):
    frames = []
    for index, row in tqdm(
        orbbec_timestamps.iterrows(),
        desc="Processing frames",
        total=len(orbbec_timestamps),
    ):
        if index % skip_n_frames != 0:
            continue
        frame_num = int(row["frame_number"])

        # [INFO] Add Hand Key-Points 3D
        hand_keypoints_aria = np.load(PATH_TO_HAND_ANNOTATIONS / f"hand_{frame_num:06d}.npy", allow_pickle=True).item()
        hand_keypoints_aria = np.array([v for k, v in hand_keypoints_aria.items() if "normal" not in k])
        # (3, N)
        hand_keypoints_aria = hand_keypoints_aria.T
        T_orbbec_aria = np.load("data/recordings/20250206_Testing/Aria/T_orbbec_aria.npy")  # 4x4 matrix
        hand_keypoints_orbbec = T_orbbec_aria @ np.concatenate(
            [hand_keypoints_aria, np.ones((1, hand_keypoints_aria.shape[1]))], axis=0
        )
        hand_keypoints_orbbec = hand_keypoints_orbbec[:3]
        hand_keypoints_orbbec = Rotation.from_euler("y", 90, degrees=True).as_matrix() @ hand_keypoints_orbbec
        hand_keypoints_aria = hand_keypoints_orbbec.T

        collage = []
        for id_, camera in input_dirs.items():
            if id_ == "Marshall":
                frame_num += 3
            elif id_ == "Orbbec":
                frame_num += 4
            multi_view_frames = camera.glob(f"color_{frame_num:06d}_camera*.jpg")
            multi_view_frames = sorted(multi_view_frames)
            images = [cv2.imread(str(frame)) for frame in multi_view_frames]

            # [INFO] Draw Hand Key-Points on the images
            for i, image in enumerate(images):
                cam_idx = f"camera{i + 1 if id_ == 'Orbbec' else i + 1 + 4:02d}"
                # [INFO] Undistort the image
                if id_ == "Marshall":
                    image = cv2.undistort(
                        image,
                        cam_infos[cam_idx]["intrinsics"],
                        np.array(
                            [cam_infos[cam_idx]["radial_params"][0]]
                            + [cam_infos[cam_idx]["radial_params"][1]]
                            + list(cam_infos[cam_idx]["tangential_params"][:2])
                            + [cam_infos[cam_idx]["radial_params"][2]]
                            + [0, 0, 0]
                        ),
                    )
                elif id_ == "Orbbec":
                    image = undistort_image(image, cam_infos[cam_idx], "color")

                if id_ == "Marshall":
                    _hand_keypoints_aria =  hand_keypoints_aria.copy() @ T_marshall_orbbec.T
                else:
                    _hand_keypoints_aria = hand_keypoints_aria.copy()

                for hand_keypoint in _hand_keypoints_aria:
                    if id_ in ["Marshall", "Orbbec"]:
                        # Project 3D point to 2D image coordinates
                        point_2d = project_to_2d(
                            hand_keypoint,
                            cam_infos[cam_idx]["intrinsics"],
                            np.linalg.inv(cam_infos[cam_idx]["extrinsics"])
                            if id_ == "Orbbec"
                            else cam_infos[cam_idx]["extrinsics"],
                        )
                        # Draw the point on the image
                        cv2.circle(
                            image,
                            (int(point_2d[0]), int(point_2d[1])),
                            10,
                            (255, 0, 0),
                            thickness=-1,
                        )

                images[i] = image

            # resize images to height 1080 if they are not already
            for i, image in enumerate(images):
                if image.shape[0] != 1080:
                    images[i] = cv2.resize(image, (1080, 1080))
            collage.extend(images)

        if len(collage) != 7:
            break
        collage = create_collage(collage, format=[3, 4], downscale_factor=2, frame_index=frame_num)
        frames.append(collage)

    create_video_from_images(frames, fps, output_path)


if __name__ == "__main__":
    recording_name = "20250206_Testing"
    cameras = ["Aria", "Marshall", "Orbbec"]

    base_input_dir = Path("data", "recordings", recording_name)
    output_dir = base_input_dir / "visualization" / "color"
    output_dir.mkdir(parents=True, exist_ok=True)

    PATH_TO_HAND_ANNOTATIONS = base_input_dir / "Aria" / "export" / "hand"
    T_marshall_orbbec = Rotation.from_euler("x", -90, degrees=True).as_matrix()

    input_dirs = {
        "Aria": base_input_dir / "Aria" / "export" / "color",
        "Marshall": base_input_dir / "Marshall" / "recording" / "export",
        "Orbbec": base_input_dir / "Orbbec" / "color",
    }

    cam_infos = load_cam_infos(base_input_dir)

    orbbec_timestamp_file = base_input_dir / "tables_timestamps.arrow"
    orbbec_timestamps = ds.dataset(orbbec_timestamp_file, format="arrow").to_table().to_pandas()

    process_video(
        cam_infos=cam_infos,
        orbbe_timestamps=orbbec_timestamps,
        input_dirs=input_dirs,
        output_path=output_dir / f"{'_'.join(cameras)}_HANDS.mp4",
        skip_n_frames=2,
        fps=12,
    )
