from pathlib import Path
import argparse
import rootutils
import cv2
from tqdm import tqdm
import pandas as pd
from itertools import chain

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from lib.utils.image import create_collage, create_video_from_images

PATH_TO_DATA = Path("data")
CAMERAS = ["Marshall", "Orbbec", "Aria"]
IDX_RANGE = [0, 15_000]

def get_args():
    parser = argparse.ArgumentParser(
        prog="Visualize Data"
    )
    parser.add_argument("--recording", "-r", required=False, default="20250227_Testing", type=str)

    args = parser.parse_args()
    return args

def get_frames_for_index(recording, index):
    
    camera_frame_dict = {}

    for camera in CAMERAS:
        frame_paths = sorted(
            (PATH_TO_DATA / "recordings" / recording / camera / "recording" / "export").glob(
                f"color_{index:06d}_camera*.jpg"
            )
        )

        camera_frame_dict[camera] = [cv2.imread(_p) for _p in frame_paths]

    return camera_frame_dict


def main(recording):
    mv_images = []
    timestamp_df = pd.read_pickle(f"data/recordings/{recording}/Marshall/recording/timestamps_camera1.pkl")

    for frame_index in tqdm(range(IDX_RANGE[0], IDX_RANGE[1], 2)):
        camera_frame_dict = get_frames_for_index(recording, frame_index)

        imgs = list(chain.from_iterable(camera_frame_dict.values()))
        if not imgs:
            continue

        mv_image = create_collage(
            imgs,
            format=None,
            downscale_factor=2,
            frame_index=f"{frame_index} | Timestamp: {timestamp_df.loc[frame_index, 'timestamp']}",
        )
        mv_images.append(mv_image)

    output_path = Path("data", "recordings", recording, "Marshall", "recording", "visualization", "unix_time.mp4")
    (output_path.parent).mkdir(exist_ok=True)
    create_video_from_images(mv_images, fps=10, output_path=output_path)

        
if __name__=='__main__':
    args = get_args()
    main(args.recording)
        