"""
Script to extract frames from MP4 videos, supporting both timestamp-based extraction
and fixed-rate (8 FPS) extraction when timestamps are not available.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import ffmpeg
import pyarrow.dataset as ds
from pandas import DataFrame as df
import numpy as np
from tqdm import tqdm

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils.camera import load_cam_infos

def get_creation_time(video_path):
    """
    Extract the creation_time metadata from an MP4 file using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        datetime: The creation_time in UTC as a datetime object.
    """
    metadata = ffmpeg.probe(video_path)

    creation_time_str = metadata["format"]["tags"]["creation_time"]
    timecode = metadata["streams"][1]["tags"]["timecode"]
    creation_time_str = creation_time_str[:11] + timecode + creation_time_str[22:]
    return datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))  # Convert to datetime object


def process_files_with_timestamps(files, timestamp_df, output_dir, cam_num, cam_params):
    """Process files when timestamp data is available."""
    timestamps = defaultdict(list)
    for video_path in files:
        creation_time = get_creation_time(video_path)
        cap = cv2.VideoCapture(str(video_path))

        video_frame_num = 0  # Initialize video frame number
        timestamp_pointer = 0  # Pointer for the timestamp DataFrame
        prev_video_timestamp = None  # To store the previous video timestamp for comparison

        progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        while cap.isOpened() and timestamp_pointer < len(timestamp_df):
            ret, frame = cap.read()
            if not ret:
                break

            # Get the video timestamp for the current frame
            relative_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            video_timestamp = (creation_time + timedelta(milliseconds=relative_timestamp)).replace(tzinfo=None)

            # Get the corresponding timestamp from the DataFrame
            df_frame = timestamp_df.iloc[timestamp_pointer]
            df_timestamp = df_frame.drop("frame_number").median()
            df_timestamp = datetime.fromtimestamp(df_timestamp / 1e9).replace(tzinfo=None)

            # If this is the first frame, or if advancing to the next video frame is better
            if prev_video_timestamp is None or abs(video_timestamp - df_timestamp) < abs(
                prev_video_timestamp - df_timestamp
            ):
                prev_frame = frame.copy()
                prev_video_timestamp = video_timestamp
                print(f"Video frame: {video_frame_num}, Video Timestamp: {video_timestamp}")
                print(f"Orbbec Frame: {timestamp_pointer}, Orbbec Timestamp: {df_timestamp}")
                print(f"Difference in nanoseconds: {abs(video_timestamp - df_timestamp)}")

                video_frame_num += 1  # Advance video pointer
            else:
                # Export the current video frame (video_frame_num should correspond to the desired timestamp in the DataFrame)
                output_frame_path = output_dir / f"color_{df_frame['frame_number']:06d}_camera{cam_num:02d}.jpg"
                # Undistort the image
                prev_frame = cv2.undistort(
                    prev_frame,
                    cam_params["K"],
                    np.array(
                        [cam_params["radial_params"][0]]
                        + [cam_params["radial_params"][1]]
                        + list(cam_params["tangential_params"][:2])
                        + [cam_params["radial_params"][2]]
                        + [0, 0, 0]
                    ),
                )
                cv2.imwrite(str(output_frame_path), prev_frame)

                # Move both pointers (video and timestamp)
                timestamp_pointer += 1
                video_frame_num += 1  # Proceed to the next video frame
                timestamps["frame_number"].append(df_frame["frame_number"])
                timestamps["timestamp"].append(prev_video_timestamp)
                timestamps["orbbec_timestamp"].append(df_timestamp)
                print(f"EXPORTED: {output_frame_path}")

                prev_frame = frame.copy()
                prev_video_timestamp = video_timestamp

            progress_bar.update(1)

        cap.release()

    return dict(timestamps)


def process_files_at_8fps(files, output_dir, cam_num, cam_params):
    """Process files at a fixed rate of 8 FPS when timestamp data is not available."""
    timestamps = defaultdict(list)
    frame_count = 0

    for video_path in files:
        creation_time = get_creation_time(video_path)
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 8)  # Calculate interval for 8 FPS
        if frame_interval < 1:
            frame_interval = 1  # Ensure we extract at least some frames

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = tqdm(total=total_frames)

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            progress_bar.update(1)

            # Process every nth frame to achieve approximately 8 FPS
            if frame_index % frame_interval == 0:
                relative_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamp = creation_time + timedelta(milliseconds=relative_timestamp)

                # Save the frame
                output_frame_path = output_dir / f"color_{frame_count:06d}_camera{cam_num:02d}.jpg"
                # Undistort the image
                frame = cv2.undistort(
                    frame,
                    cam_params["K"],
                    np.array(
                        [cam_params["radial_params"][0]]
                        + [cam_params["radial_params"][1]]
                        + list(cam_params["tangential_params"][:2])
                        + [cam_params["radial_params"][2]]
                        + [0, 0, 0]
                    ),
                )
                cv2.imwrite(str(output_frame_path), frame)

                # Store the timestamp information
                timestamps["frame_number"].append(frame_count)
                # Export the timestamp in nanoseconds
                timestamps["timestamp"].append(int(frame_timestamp.timestamp() * 1e9))

                print(f"EXPORTED 15FPS: {output_frame_path}, Timestamp: {frame_timestamp}")

                frame_count += 1

            frame_index += 1

        cap.release()

    return dict(timestamps)


def extract_frames_with_timestamps(input_dir, output_dir, timestamps, cam_infos):
    """
    Extract frames from MP4 videos, save them as JPG files, and export timestamps with Unix time.

    Args:
        input_dir (Path): Directory containing the MP4 files.
        output_dir (Path): Directory where frames and timestamp CSV will be saved.
        timestamps: DataFrame with timestamps or None for 15 FPS extraction.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files for each camera
    ch1_files = sorted(input_dir.glob("*ch1*.mp4"))
    ch2_files = sorted(input_dir.glob("*ch2*.mp4"))

    if timestamps is not None:
        # Process with timestamp synchronization
        print("Processing with timestamp synchronization...")
        timestamps_array_1 = process_files_with_timestamps(ch1_files, timestamps, output_dir, cam_num=1, cam_params=cam_infos["camera05"])
        timestamps_array_2 = process_files_with_timestamps(
            ch2_files, timestamps, output_dir, cam_num=2, cam_params=cam_infos["camera06"]
        )
    else:
        # Process at 8 FPS
        print("Processing at 8 FPS (no timestamps file provided)...")
        timestamps_array_1 = process_files_at_8fps(ch1_files, output_dir, cam_num=1, cam_params=cam_infos["camera_05"])
        timestamps_array_2 = process_files_at_8fps(ch2_files, output_dir, cam_num=2, cam_params=cam_infos["camera_06"])

    # Save timestamps as pandas DataFrame
    timestamps_array_1 = df(timestamps_array_1)
    timestamps_array_2 = df(timestamps_array_2)

    timestamps_array_1.to_pickle(output_dir / "timestamps_camera1.pkl")
    timestamps_array_2.to_pickle(output_dir / "timestamps_camera2.pkl")


def main():
    recording_name = "20250227_Testing"  # Replace with your recording name
    base_dir = Path("data/recordings") / recording_name / "Marshall"
    input_dir = base_dir / "recording" / "videos"
    output_dir = base_dir / "recording" / "export"

    timestamps_file = Path("data") / "recordings" / recording_name / "tables_timestamps.arrow"

    # Read an Arrow file
    if timestamps_file.exists():
        print(f"Reading timestamps from {timestamps_file}")
        timestamps = ds.dataset(timestamps_file, format="arrow").to_table().to_pandas()
    else:
        print(f"Timestamp file not found: {timestamps_file}")
        print("Exporting with 8 fps")
        timestamps = None

    # Load the camera parameters
    cam_infos = load_cam_infos(Path("data/recordings") / recording_name)

    extract_frames_with_timestamps(input_dir, output_dir, timestamps, cam_infos)


if __name__ == "__main__":
    main()
