#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import cv2
import numpy as np
from typing import List, Tuple, Optional
import random
import json
import math
from scipy.spatial.transform import Rotation as R
from aruco_ops import detect_marker

# Constants
COLORS = [
    (0, 0, 255),    # red
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (255, 255, 0),  # cyan
]
MARKER_SIZE = 0.54  # Marker size in meters
POINTS_3D = np.array([
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
])
ORIGIN_3D = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # Origin point for projection

def get_camera_images(base_path: Path, camera_num: int) -> list[Path]:
    """Get all camera images for the specified camera number."""
    pattern = f"*_camera{camera_num:02d}.jpg"
    return sorted(base_path.glob(pattern))

def draw_axes(img: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
               rvec: np.ndarray, tvec: np.ndarray, length: float = 0.3) -> np.ndarray:
    """Draw 3D coordinate axes on the image."""
    axis_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,length]])
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    
    # Draw the axes
    origin = tuple(map(int, img_points[0].ravel()))
    x_point = tuple(map(int, img_points[1].ravel()))
    y_point = tuple(map(int, img_points[2].ravel()))
    z_point = tuple(map(int, img_points[3].ravel()))
    
    img = cv2.line(img, origin, x_point, (0,0,255), 3)  # X axis: Red
    img = cv2.line(img, origin, y_point, (0,255,0), 3)  # Y axis: Green
    img = cv2.line(img, origin, z_point, (255,0,0), 3)  # Z axis: Blue
    
    return img

def process_image(
    image_path: Path, 
    output_path: Path = None, 
    camera_matrix: Optional[np.ndarray] = None, 
    dist_coeffs: Optional[np.ndarray] = None,) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Process a single image by detecting markers and drawing circles."""
    img = cv2.imread(str(image_path))
    points = detect_marker(str(image_path))
    
    points_2d = None
    if len(points) == len(POINTS_3D):
        points_2d = np.array([(point.x, point.y) for point in points], dtype=np.float32)
    
    for i, point in enumerate(points):
        cv2.circle(img, (int(point.x), int(point.y)), 5, COLORS[i % len(COLORS)], thickness=-1)
    
    if output_path and camera_matrix is not None and dist_coeffs is not None and points_2d is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process undistorted image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        undistorted_path = output_path.parent / f"undistorted_{output_path.name}" 
        cv2.imwrite(str(undistorted_path), undistorted_img)
    
    return img, points_2d

def calibrate_camera(images_2d: List[np.ndarray], num_calibration_images: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate camera using detected 2D points and known 3D points."""
    if len(images_2d) > num_calibration_images:
        selected_indices = random.sample(range(len(images_2d)), num_calibration_images)
        images_2d = [images_2d[i] for i in selected_indices]
    
    object_points = [POINTS_3D.astype(np.float32) for _ in range(len(images_2d))]
    height, width = 1080, 1920
    
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        images_2d,
        (width, height),
        None, 
        None
    )
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs

def save_calibration_json(camera_num: int, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                         rvec: np.ndarray, tvec: np.ndarray, output_dir: Path):
    """Save calibration results in the requested JSON format."""
    # Convert rotation vector to rotation matrix
    rot_matrix, _ = cv2.Rodrigues(rvec)
    
    # Convert rotation matrix to quaternion
    quat = R.from_matrix(rot_matrix).as_quat()
    
    # Get image dimensions
    height, width = 1080, 1920 # Update these values if different
    
    # Calculate metric radius (you may want to adjust this calculation)
    metric_radius = np.sqrt(width*width + height*height) / 2.0
    
    # Create calibration data structure
    calibration_data = {
        "value0": {
            "color_resolution": 6,  # HD mode
            "color_parameters": {
                "fov_x": float(camera_matrix[0, 0]),
                "fov_y": float(camera_matrix[1, 1]),
                "c_x": float(camera_matrix[0, 2]),
                "c_y": float(camera_matrix[1, 2]),
                "width": width,
                "height": height,
                "intrinsics_matrix": {
                    "m00": float(camera_matrix[0, 0]),
                    "m10": 0.0,
                    "m20": float(camera_matrix[0, 2]),
                    "m01": 0.0,
                    "m11": float(camera_matrix[1, 1]),
                    "m21": float(camera_matrix[1, 2]),
                    "m02": 0.0,
                    "m12": 0.0,
                    "m22": 1.0
                },
                "radial_distortion": {
                    "m00": float(dist_coeffs[0]),
                    "m10": float(dist_coeffs[1]),
                    "m20": float(dist_coeffs[4]),
                    "m30": 0.0,
                    "m40": 0.0,
                    "m50": 0.0
                },
                "tangential_distortion": {
                    "m00": float(dist_coeffs[2]),
                    "m10": float(dist_coeffs[3])
                },
                "metric_radius": float(metric_radius)
            },
            "camera_pose": {
                "translation": {
                    "m00": float(tvec[0]),
                    "m10": float(tvec[1]),
                    "m20": float(tvec[2])
                },
                "rotation": {
                    "x": float(quat[0]),
                    "y": float(quat[1]),
                    "z": float(quat[2]),
                    "w": float(quat[3])
                }
            },
            "is_valid": True
        }
    }
    
    # Save to JSON file
    output_file = output_dir / f"camera_params_{camera_num:02d}.json"
    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    print(f"Saved camera parameters to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Process camera images for Aruco marker detection and calibration')
    parser.add_argument('--camera_num', default=1, type=int, help='Camera number to process')
    parser.add_argument('--viz', default=True, action='store_true', help='Save debug visualizations')
    parser.add_argument('--num-images', type=int, default=200,
                       help='Number of images to use for calibration (default: 20)')
    args = parser.parse_args()
    
    recording = "20250206_Testing"
    
    base_path = Path("data") / "recordings" / recording / "Marshall" / "calibration" / "export"
    image_paths = get_camera_images(base_path, args.camera_num)
    
    if not image_paths:
        print(f"No images found for camera {args.camera_num}")
        return
    
    valid_points_2d = []
    camera_matrix = None
    dist_coeffs = None
    
    # First pass to collect valid points for calibration
    for image_path in image_paths:
        print(f"Processing {image_path.name} (first pass)")
        _, points_2d = process_image(image_path)
        
        if points_2d is not None:
            valid_points_2d.append(points_2d)
    
    print(f"\nFound {len(valid_points_2d)} images with all points detected")
    
    if len(valid_points_2d) < args.num_images:
        print(f"Warning: Only found {len(valid_points_2d)} valid images, "
              f"but {args.num_images} were requested for calibration")
        print("Proceeding with all available valid images")
    
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(valid_points_2d, args.num_images)
    
    # Refine pose using iterative LM optimization
    success, rvec, tvec = cv2.solvePnP(
        POINTS_3D, 
        valid_points_2d[0], 
        camera_matrix, 
        dist_coeffs,
        rvec=rvecs[0], 
        tvec=tvecs[0], 
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Second pass to create visualization with origin projection
    if args.viz:
        for i, image_path in enumerate(image_paths):
            print(f"Processing {image_path.name} (second pass with origin projection)")
            output_path = Path("output/debug/marshall") / image_path.name
            # Only project origin for images that were used in calibration
            if i < len(rvecs):
                process_image(image_path, output_path, camera_matrix, dist_coeffs)
    
    # Save results
    output_dir = Path("output/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save traditional NPY files
    np.save(output_dir / f"camera_matrix_{args.camera_num:02d}.npy", camera_matrix)
    np.save(output_dir / f"dist_coeffs_{args.camera_num:02d}.npy", dist_coeffs)

    # p = cv2.projectPoints(POINTS_3D, rvec, tvec, camera_matrix, dist_coeffs)
    # img = cv2.imread(str(image_paths[0]))
    # for point in p[0]:
    #     cv2.circle(img, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), thickness=-1)
    
    # Save JSON format with quaternions (using first frame's pose)
    save_calibration_json(args.camera_num, camera_matrix, dist_coeffs[0], 
                         rvec, tvec, output_dir)
    
    print("\nCalibration Results:")
    print("Camera Matrix:")
    print(camera_matrix)
    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())
    print(f"\nRMS error: {rms}")

if __name__ == "__main__":
    main()