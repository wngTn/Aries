import math
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
import open3d as o3d
import tqdm


def point_cloud_from_images(
    color_image_path: Union[str, Path], depth_image_path: Union[str, Path], cam_infos: Dict[str, Dict], idx: int = 0
) -> List[o3d.geometry.PointCloud]:
    """
    Generate point clouds from color and depth images using camera information.

    This function processes pairs of color and depth images along with camera
    calibration data to create 3D point clouds. It handles multiple cameras and
    applies the appropriate transformations based on camera parameters.

    Args:
        color_image_path: Path to the color image folder
        depth_image_path: Path to the depth image folder
        cam_infos: Dictionary containing camera calibration information
            Expected format:
            {
                'camera01': {
                    'width': int,
                    'height': int,
                    'fov_x': float,
                    'fov_y': float,
                    'c_x': float,
                    'c_y': float,
                    'extrinsics': 4x4 transformation matrix
                },
                ...
            }
        idx: Index of the image pair to process (default: 4078)

    Returns:
        List of Open3D PointCloud objects, one for each camera

    Raises:
        ValueError: If camera information is missing or invalid
        FileNotFoundError: If image files cannot be found

    Note:
        - Depth values are assumed to be in millimeters and are converted to meters
        - The depth truncation is set to 10000000 units
        - RGB values are preserved (not converted to intensity)
    """
    point_clouds = []

    for cam_idx in range(1, len(cam_infos) + 1):
        camera_key = f"camera0{cam_idx}"

        try:
            cam_info = cam_infos[camera_key]
        except KeyError:
            raise ValueError(f"Missing camera information for {camera_key}")

        # Create camera intrinsics object
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=cam_info["width"],
            height=cam_info["height"],
            fx=cam_info["fov_x"],
            fy=cam_info["fov_y"],
            cx=cam_info["c_x"],
            cy=cam_info["c_y"],
        )

        # Load and convert images to Open3D format
        rgb_image_path = color_image_path / f"color_{idx:06d}_camera{cam_idx:02d}.jpg"
        rgb_image = cv2.imread(str(rgb_image_path))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # undistort rgb image
        rgb_image = undistort_image(rgb_image, cam_info, modality="color")
        rgb_o3d = o3d.geometry.Image(np.asarray(rgb_image))

        d_image_path = depth_image_path / f"depth_{idx:06d}_camera{cam_idx:02d}.tiff"
        depth_image = cv2.imread(str(d_image_path), cv2.IMREAD_ANYDEPTH)
        # TODO this is wrong
        if depth_image.shape != rgb_image.shape[:2]:
            depth_image = cv2.resize(
                depth_image, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        # undistort depth image
        depth_image = undistort_image(depth_image, cam_info, modality="depth")
        depth_o3d = o3d.geometry.Image(np.asarray(depth_image))

        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=rgb_o3d, depth=depth_o3d, convert_rgb_to_intensity=False, depth_scale=1, depth_trunc=10000000
        )

        # Generate point cloud
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_image, intrinsic=intrinsics)

        # Convert points from millimeters to meters
        points_array = np.asarray(point_cloud.points) / 1000
        point_cloud.points = o3d.utility.Vector3dVector(points_array)

        # Apply extrinsic transformation
        point_cloud.transform(cam_info["extrinsics"])

        point_clouds.append(point_cloud)

    return point_clouds


def undistort_image(img, cam_params, modality=str):
    """
    Undistorts an image using the provided camera parameters.
    Parameters:
    img (numpy.ndarray): The input image to be undistorted.
    cam_params (dict): A dictionary containing camera parameters with the following keys:
        - 'intrinsics' (numpy.ndarray): The camera intrinsic matrix.
        - 'radial_params' (list): Radial distortion coefficients.
        - 'tangential_params' (list): Tangential distortion coefficients.
        - 'width' (int): The width of the image.
        - 'height' (int): The height of the image.
    modality (str): The modality of the image, either 'depth' or 'color'. Determines the interpolation and border mode.
    Returns:
    numpy.ndarray: The undistorted image.
    Raises:
    ValueError: If the modality is not 'depth' or 'color'.
    """
    if modality == "depth":
        interpolation = cv2.INTER_NEAREST
        borderMode = cv2.BORDER_CONSTANT
    elif modality == "color":
        interpolation = cv2.INTER_LINEAR
        borderMode = cv2.BORDER_TRANSPARENT
    else:
        raise ValueError("Invalid modality. Must be 'depth' or 'color'")

    K = cam_params["intrinsics"]
    distortion_coeffs = np.array(
        cam_params["radial_params"][:2] + cam_params["tangential_params"] + cam_params["radial_params"][2:]
    )

    # Adjusted alpha parameter to 1
    newCameraMatrix, _ = cv2.getOptimalNewCameraMatrix(
        K, distortion_coeffs, (cam_params["width"], cam_params["height"]), 1
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        K, distortion_coeffs, None, newCameraMatrix, (cam_params["width"], cam_params["height"]), cv2.CV_32FC1
    )

    undistorted_image = cv2.remap(img, map1, map2, interpolation=interpolation, borderMode=borderMode)

    return undistorted_image


def create_collage(images, format=None, downscale_factor=2, frame_index=None):
    # First, downscale each image
    downscaled_images = [
        cv2.resize(img, (img.shape[1] // downscale_factor, img.shape[0] // downscale_factor)) for img in images
    ]

    # Validate format
    if format is None:
        # Default to original behavior: split into two rows
        num_images = len(downscaled_images)
        format = [(num_images + 1) // 2, num_images // 2]
    else:
        if math.prod(format) < len(images):
            raise ValueError(f"Format {format} sum ({sum(format)}) doesn't match number of images ({len(images)})")

    # Get the max width and height for uniform grid
    max_width = max(img.shape[1] for img in downscaled_images)
    max_height = max(img.shape[0] for img in downscaled_images)

    # Create canvas with appropriate dimensions
    num_rows = len(format)
    max_images_per_row = max(format)
    canvas_height = num_rows * max_height
    canvas_width = max_images_per_row * max_width
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place images row by row
    current_img_idx = 0
    for row_idx, num_images_in_row in enumerate(format):
        # Calculate total width of images in this row
        row_images = downscaled_images[current_img_idx : current_img_idx + num_images_in_row]
        row_width = sum(img.shape[1] for img in row_images)

        # Center the row
        x_offset = (canvas_width - row_width) // 2
        y_offset = row_idx * max_height

        # Place images in the row
        for img in row_images:
            h, w, _ = img.shape
            canvas[y_offset : y_offset + h, x_offset : x_offset + w] = img
            x_offset += w

        current_img_idx += num_images_in_row

    # Add frame index if provided
    if frame_index is not None:
        cv2.putText(
            canvas,
            f"Frame {frame_index}",
            (10, canvas_height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def create_video_from_images(images, fps, output_path):
    # Check if the list of images is empty
    if not images:
        return

    # Get the height and width of the first image
    height, width = images[0].shape[:2]

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in tqdm.tqdm(images, desc="Creating video"):
        if img.shape[:2] != (height, width):
            raise ValueError("All images must have the same dimensions")
        video_writer.write(img)

    video_writer.release()
    print(f"Video saved successfully at {output_path}")


def draw_mano_2d(image, keypoints, thickness=2, radius=5):
    """
    Visualize MANO hand keypoints on an image.

    Args:
        image: Input RGB image (numpy array)
        keypoints: Array of shape (2, 16, 3) where:
                  - First dimension: 0 for left hand, 1 for right hand
                  - Second dimension: 16 keypoints per hand
                  - Third dimension: x, y, confidence/visibility
        thickness: Thickness of the connection lines
        radius: Radius of the keypoint circles

    Returns:
        Annotated image with hand keypoints and connections
    """
    # Make a copy of the image to avoid modifying the original
    img_out = image.copy()

    # Define colors for each keypoint (BGR format)
    colors = [
        (255, 0, 0),  # Blue - Wrist
        (0, 255, 0),  # Green - Thumb base
        (0, 255, 255),  # Yellow - Thumb mid
        (0, 0, 255),  # Red - Thumb tip
        (255, 0, 255),  # Magenta - Index base
        (255, 128, 0),  # Light blue - Index mid
        (255, 255, 0),  # Cyan - Index tip
        (128, 0, 255),  # Purple - Middle base
        (0, 128, 255),  # Orange - Middle mid
        (128, 255, 0),  # Light green - Middle tip
        (128, 128, 0),  # Dark yellow - Ring base
        (0, 128, 128),  # Dark cyan - Ring mid
        (0, 255, 128),  # Teal - Ring tip
        (128, 0, 128),  # Dark magenta - Pinky base
        (64, 0, 255),  # Pink - Pinky mid
        (255, 64, 0),  # Light red - Pinky tip
    ]

    # Define connections between keypoints
    # Each tuple represents (start_point_idx, end_point_idx)
    connections = [
        (0, 1),  # Wrist to thumb base
        (1, 2),  # Thumb base to thumb mid
        (2, 3),  # Thumb mid to thumb tip
        (0, 4),  # Wrist to index base
        (4, 5),  # Index base to index mid
        (5, 6),  # Index mid to index tip
        (0, 7),  # Wrist to middle base
        (7, 8),  # Middle base to middle mid
        (8, 9),  # Middle mid to middle tip
        (0, 10),  # Wrist to ring base
        (10, 11),  # Ring base to ring mid
        (11, 12),  # Ring mid to ring tip
        (0, 13),  # Wrist to pinky base
        (13, 14),  # Pinky base to pinky mid
        (14, 15),  # Pinky mid to pinky tip
    ]

    # Make sure the input has the correct shape
    assert keypoints.shape == (2, 16, 2), f"Expected shape (2, 16, 3), got {keypoints.shape}"

    # Process each hand (left and right)
    hand_names = ["Left", "Right"]
    for hand_idx in range(2):
        # Check if hand is present (if all values are 0, skip this hand)
        if np.all(keypoints[hand_idx] == 0):
            continue

        # Extract 2D coordinates for this hand
        hand_keypoints = keypoints[hand_idx, :, :2].astype(np.int32)

        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start_point = tuple(hand_keypoints[start_idx])
            end_point = tuple(hand_keypoints[end_idx])

            # Draw the line - different shade for left/right hand
            line_color = (102, 102, 102) if hand_idx == 0 else (50, 50, 200)
            cv2.line(img_out, start_point, end_point, line_color, thickness)

        # Draw keypoints on top of the lines
        for i, (x, y) in enumerate(hand_keypoints):
            # Use slightly different colors for left/right hand
            color = colors[i]
            if hand_idx == 1:  # Right hand - adjust color to distinguish
                color = tuple(max(0, c - 50) for c in colors[i])

            cv2.circle(img_out, (x, y), radius, color, -1)  # -1 means filled circle

            # Optionally add keypoint label
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(img_out, f"{i}", (x+5, y-5), font, 0.5, (255, 255, 255), 1)

    return img_out