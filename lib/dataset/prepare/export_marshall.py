import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import ffmpeg

import cv2
import pyarrow.dataset as ds
from tqdm import tqdm

from collections import defaultdict
from pandas import DataFrame as df


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
    return datetime.fromisoformat(
        creation_time_str.replace("Z", "+00:00")
    )  # Convert to datetime object


def process_files(files, timestamp_df, output_dir, cam_num):
    timestamps = defaultdict(list)
    for video_path in files:
        creation_time = get_creation_time(video_path)
        cap = cv2.VideoCapture(str(video_path))

        video_frame_num = 0  # Initialize video frame number
        timestamp_pointer = 0  # Pointer for the timestamp DataFrame
        prev_video_timestamp = (
            None  # To store the previous video timestamp for comparison
        )

        progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        while cap.isOpened() and timestamp_pointer < len(timestamp_df):
            ret, frame = cap.read()
            if not ret:
                break

            # Get the video timestamp for the current frame
            relative_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            video_timestamp = (
                creation_time + timedelta(milliseconds=relative_timestamp)
            ).replace(tzinfo=None)

            # Get the corresponding timestamp from the DataFrame
            df_frame = timestamp_df.iloc[timestamp_pointer]
            df_timestamp = df_frame.drop("frame_number").median()
            df_timestamp = datetime.fromtimestamp(df_timestamp / 1e9).replace(
                tzinfo=None
            )

            # If this is the first frame, or if advancing to the next video frame is better
            if prev_video_timestamp is None or abs(video_timestamp - df_timestamp) < abs(prev_video_timestamp - df_timestamp):
                prev_frame = frame.copy()
                prev_video_timestamp = video_timestamp
                print(f"Video frame: {video_frame_num}, Video Timestamp: {video_timestamp}")
                print(
                    f"Orbbec Frame: {timestamp_pointer}, Orbbec Timestamp: {df_timestamp}"
                )
                print(f"Difference in nanoseconds: {abs(video_timestamp - df_timestamp)}")
                
                video_frame_num += 1  # Advance video pointer
            else:
                # Export the current video frame (video_frame_num should correspond to the desired timestamp in the DataFrame)
                output_frame_path = output_dir / f"color_{df_frame['frame_number']:06d}_camera{cam_num:02d}.jpg"
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


def extract_frames_with_timestamps(input_dir, output_dir, timestamps):
    """
    Extract frames from MP4 videos, save them as JPG files, and export timestamps with Unix time.

    Args:
        input_dir (Path): Directory containing the MP4 files.
        output_dir (Path): Directory where frames and timestamp CSV will be saved.
        frame_skip (int): Number of frames to skip between each saved frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process files for each camera
    ch1_files = sorted(input_dir.glob("*ch1*.mp4"))
    ch2_files = sorted(input_dir.glob("*ch2*.mp4"))

    timestamps_array_1 = process_files(ch1_files, timestamps, output_dir, cam_num=1)
    timestamps_array_2 = process_files(ch2_files, timestamps, output_dir, cam_num=2)

    # Save timestamps as pandas DataFrame
    timestamps_array_1 = df(timestamps_array_1)
    timestamps_array_2 = df(timestamps_array_2)
    
    timestamps_array_1.to_pickle(output_dir / "timestamps_camera1.pkl")
    timestamps_array_2.to_pickle(output_dir / "timestamps_camera2.pkl")
    


def main():
    recording_name = "20250206_Testing"  # Replace with your recording name
    base_dir = Path("data/recordings") / recording_name / "Marshall"
    input_dir = base_dir / "recording" / "videos"
    output_dir = base_dir / "recording" / "export"

    timestamps_file = (
        Path("data") / "recordings" / recording_name / "tables_timestamps.arrow"
    )

    # Read an Arrow file
    timestamps = ds.dataset(timestamps_file, format="arrow").to_table().to_pandas()

    extract_frames_with_timestamps(input_dir, output_dir, timestamps)


if __name__ == "__main__":
    main()
