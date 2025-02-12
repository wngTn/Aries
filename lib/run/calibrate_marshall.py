import argparse
import sys
import os

sys.path.append(
    "/Users/tonywang/Documents/hiwi/Aries/Code/Aries/lib/calibration/aruco-3.1.12/build/utils_calibration"
)

# import of the calibration module
import calibration_module

# arguments control
parser = argparse.ArgumentParser(description="Calibration with aruco markers")
parser.add_argument("path_to_images", type=str, help="Path to images")
parser.add_argument("output_json", type=str, help="Name of output json file")
parser.add_argument("marker_length", type=int, help="Marker length in meters")

args = parser.parse_args()


calibration_module.calibrate_camera(args.path_to_images, args.output_json, args.marker_length, 40)
