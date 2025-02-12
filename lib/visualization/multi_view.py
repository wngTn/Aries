import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

import rootutils
rootutils.setup_root(__file__, ".project-root", pythonpath=True, dotenv=True)

from lib.utils.image import create_collage, create_video_from_images



def process_video(input_dir, output_path, start_frame=0, end_frame=None, skip_n_frames=1, fps=30):
    # Get all camera folders
    pattern = os.path.join(input_dir, "color_*_camera*.jpg")
    all_files = sorted(glob.glob(pattern))
    
    # Group files by frame number
    frame_groups = {}
    for file_path in all_files:
        frame_num = int(os.path.basename(file_path).split('_')[1])
        if frame_num not in frame_groups:
            frame_groups[frame_num] = []
        frame_groups[frame_num].append(file_path)
    
    # Get sorted frame numbers and apply frame range
    frame_numbers = sorted(frame_groups.keys())
    if end_frame is None:
        end_frame = max(frame_numbers)
    
    frame_numbers = [f for f in frame_numbers if start_frame <= f <= end_frame and (f - start_frame) % skip_n_frames == 0]
    
    # Process frames and create collages
    collages = []
    for frame_num in tqdm(frame_numbers, desc="Processing frames"):
        # Read all images for this frame
        images = [cv2.imread(file_path) for file_path in sorted(frame_groups[frame_num])]
        # Create collage
        collage = create_collage(images, downscale_factor=2, frame_index=frame_num)
        collages.append(collage)
    
    # Create video from collages
    create_video_from_images(collages, fps, output_path)

if __name__ == "__main__":
    input_dir = "data/recordings/20241023_Christian/Marshall/export"
    output_path = "output.mp4"
    
    process_video(
        input_dir=input_dir,
        output_path=output_path,
        start_frame=3820, # 600,
        end_frame= 3820 + (100 * 20), # 1_000,  # Set to None for all frames
        skip_n_frames=1,  # Process every 2nd frame
        fps=6
    )