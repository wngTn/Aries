import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
import argparse

import cv2


def get_creation_time(video_path):
    """
    Extract the creation_time metadata from an MP4 file using ffprobe.

    Args:
        video_path (str): Path to the video file.

    Returns:
        datetime: The creation_time in UTC as a datetime object.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_entries",
        "format_tags=creation_time",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    metadata = json.loads(result.stdout)

    creation_time_str = metadata["format"]["tags"]["creation_time"]
    return datetime.fromisoformat(creation_time_str.replace("Z", "+00:00"))  # Convert to datetime object


def extract_frames_with_metadata(input_dir, output_dir, frame_skip=1):
    """
    Extract frames from MP4 videos, save them as JPG files, and export timestamps with Unix time.

    Args:
        input_dir (Path): Directory containing the MP4 files.
        output_dir (Path): Directory where frames and timestamp CSV will be saved.
        frame_skip (int): Number of frames to skip between each saved frame.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_file = output_dir / "timestamps_with_metadata.csv"

    with open(timestamp_file, mode="w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Frame", "Camera", "Relative Timestamp (ms)", "Unix Timestamp (s)"])

        def process_files(files, camera_id):
            frame_count = 0
            for video_path in files:
                creation_time = get_creation_time(video_path)  # Get creation_time as a datetime object
                cap = cv2.VideoCapture(str(video_path))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Get relative timestamp in milliseconds
                    relative_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds

                    # Compute Unix timestamp
                    unix_timestamp = creation_time.timestamp() + relative_timestamp

                    # Save every (frame_skip + 1)th frame
                    if frame_count % frame_skip == 0:
                        output_path = output_dir / f"color_{frame_count:06d}_camera{camera_id:02d}.jpg"
                        cv2.imwrite(str(output_path), frame)

                        # Write timestamps to CSV
                        csv_writer.writerow(
                            [
                                frame_count,
                                f"camera{camera_id:02d}",
                                round(relative_timestamp, 3),
                                round(unix_timestamp, 3),
                            ]
                        )

                    frame_count += 1
                cap.release()

        # Process files for each camera
        ch1_files = sorted(input_dir.glob("*ch1*.mp4"))
        ch2_files = sorted(input_dir.glob("*ch2*.mp4"))

        process_files(ch1_files, 1)
        process_files(ch2_files, 2)


def main(recording_name):
    base_dir = Path("data/recordings") / recording_name / "Marshall"
    input_dir = base_dir / "calibration" / "videos"
    output_dir = base_dir / "calibration" / "export"

    extract_frames_with_metadata(input_dir, output_dir, frame_skip=20)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract frames from MP4 videos and save with metadata.")
    parser.add_argument("--recording", "-r", type=str, required=True, help="Name of the recording")
    args = parser.parse_args()
    main(args.recording)
