import numpy as np
import cv2
import json
import glob
import os
from datetime import datetime
from pathlib import Path


class CameraCalibrator:
    def __init__(self, aruco_dict=cv2.aruco.DICT_ARUCO_MIP_36H12, marker_length=0.54):
        """
        Initialize the camera calibrator for Aruco board.
        
        Args:
            aruco_dict: The Aruco dictionary type.
            marker_length (float): Size of each marker in meters.
        """
        # Initialize Aruco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
        
        # Define the board for calibration (assuming a grid board here)
        self.board = cv2.aruco.GridBoard(
            size=(5, 5),  # Number of markers in X and Y directions
            markerLength=marker_length,  # Size of each marker in meters
            markerSeparation=0.1,  # Separation between markers in meters
            dictionary=self.aruco_dict
        )
        
        # Arrays to store object points and image points
        self.all_counters = []
        self.all_corners = []
        self.all_ids = []
        
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
        # Image size for calibration
        self.image_size = None

    def process_images_from_folder(self, folder_path):
        """
        Process images from a specified folder for calibration using Aruco board.
        
        Args:
            folder_path (str): Path to the folder containing images.
        """
        image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + \
                      glob.glob(os.path.join(folder_path, '*.png')) + \
                      glob.glob(os.path.join(folder_path, '*.jpeg'))
        
        if not image_files:
            print("No images found in the specified folder.")
            return False
        
        for i, img_file in enumerate(image_files):
            image = cv2.imread(img_file)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            if corners:
                self.all_corners.append(corners)
                self.all_ids.append(ids)
                self.all_counters.append(len(ids))
                
                # Draw detected markers on the image
                image = cv2.aruco.drawDetectedMarkers(image, corners, ids)
                cv2.imwrite(f"detected_markers_{i}.jpg", image)
        
        self.all_corners = np.vstack(self.all_corners)[:, 0]
        self.all_ids = np.vstack(self.all_ids)
        self.all_counters = np.array(self.all_counters)
        print(f"Found {len(np.unique(self.all_ids))} unique markers in {len(self.all_corners)} images.")
        
        return True
    
    def calibrate(self):
        """
        Perform camera calibration using collected frames.
        
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        
        print("Performing calibration...")
        # Provide initial guesses for cameraMatrix and distCoeffs
        camera_matrix = np.eye(3)  # Identity matrix as initial guess
        dist_coeffs = np.zeros((5, 1))  # Zero distortion coefficients
        
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.aruco.calibrateCameraArucoExtended(
            corners=self.all_corners,
            ids=self.all_ids,
            counter=self.all_counters,
            board=self.board, 
            imageSize=self.image_size, 
            cameraMatrix=camera_matrix, 
            distCoeffs=dist_coeffs
        )

        
        if not ret:
            print("Calibration failed")
            return False
        
        print(f"Calibration complete! Re-projection error: {ret}")
        return True


    # The rest of the functions (save_calibration, load_calibration, etc.) can remain the same.
    
def main():
    # Create output directory for calibration files
    output_dir = Path("camera_calibration")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize calibrator
    print("Starting camera calibration using Aruco board...")
    calibrator = CameraCalibrator()
    
    folder_path = "data/Aries/recordings/20241023_Christian/calibration_raw/marshall/marshall_02"
    calibrator.process_images_from_folder(folder_path)
    calibrator.calibrate()
    # Optionally save calibration data if needed

if __name__ == "__main__":
    main()
