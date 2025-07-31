"""
Aria frame extraction and processing module.

This module provides functionality to extract and process frames from Aria device recordings,
including timestamp synchronization and image transformation operations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import rootutils
from PIL import Image
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
)
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId
from tqdm import tqdm

rootutils.setup_root(__file__, ".project-root", pythonpath=True)


@dataclass
class ProcessingConfig:
    """Configuration for frame processing parameters."""

    aria_name: str
    recording_name: str
    base_dir: Path
    devignetting_mask_path: Path
    output_image_size: Tuple[int, int] = (1024, 1024)
    focal_length: float = 300.0
    camera_label: str = "camera-rgb"

    @property
    def mps_path(self) -> Path:
        """Get the MPS path for the recording."""
        return self.base_dir / f"mps_{self.aria_name}_vrs"

    @property
    def vrs_file(self) -> Path:
        """Get the VRS file path for the recording."""
        return self.base_dir / f"{self.aria_name}.vrs"


class AriaFrameProcessor:
    """Handles extraction and processing of frames from Aria device recordings."""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize the frame processor.

        Args:
            config: Processing configuration parameters
        """
        self.config = config
        self.provider = self._initialize_provider()
        self.mps_data_provider = self._initialize_mps_provider()
        self.device_calib = self._initialize_calibration()
        self.point_cloud = self._get_point_cloud()

    def _initialize_provider(self) -> data_provider.VrsDataProvider:
        """Initialize the VRS data provider."""
        if not self.config.vrs_file.exists():
            raise FileNotFoundError(f"VRS file not found: {self.config.vrs_file}")
        return data_provider.create_vrs_data_provider(str(self.config.vrs_file))

    def _initialize_mps_provider(self) -> mps.MpsDataProvider:
        """Initialize the MPS data provider."""
        paths_provider = mps.MpsDataPathsProvider(str(self.config.mps_path))
        return mps.MpsDataProvider(paths_provider.get_data_paths())

    def _initialize_calibration(self) -> calibration.DeviceCalibration:
        """Initialize device calibration with devignetting masks."""
        device_calib = self.provider.get_device_calibration()
        device_calib.set_devignetting_mask_folder_path(str(self.config.devignetting_mask_path))
        return device_calib

    def _get_point_cloud(self):
        """Get point cloud data from the VRS file."""
        points = self.mps_data_provider.get_semidense_point_cloud()
        points = filter_points_from_confidence(points, threshold_invdep=0.01, threshold_dep=0.002)
        # Retrieve point position
        point_cloud = np.stack([it.position_world for it in points])
        return point_cloud

    def process_frames(self, output_dirs: Dict[str, Path]) -> None:
        """
        Process and extract frames using synchronized timestamps.

        Args:
            output_dirs: Dictionary mapping camera labels to output directories
        """
        self._ensure_output_dirs(output_dirs)
        timestamps = self._get_synchronized_timestamps()
        self._extract_frames(timestamps, output_dirs)

    def _ensure_output_dirs(self, output_dirs: Dict[str, Path]) -> None:
        """Create output directories if they don't exist."""
        for output_dir in output_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)

    def _get_synchronized_timestamps(self) -> pd.DataFrame:
        """Get synchronized timestamps between Aria and Orbecc devices."""
        orbecc_timestamps = self._load_orbecc_timestamps()
        aria_timestamps = self._load_aria_timestamps()
        return self._synchronize_timestamps(aria_timestamps, orbecc_timestamps)

    def _load_orbecc_timestamps(self) -> pd.DataFrame:
        """Load Orbecc timestamps from Arrow file."""
        timestamps_file = (
            Path("data") / "recordings" / self.config.recording_name / "Orbbec" / "tables_timestamps.arrow"
        )
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Orbecc timestamps file not found: {timestamps_file}")
        return ds.dataset(timestamps_file, format="arrow").to_table().to_pandas()

    def _load_aria_timestamps(self) -> pd.DataFrame:
        """Load Aria timestamps from CSV file."""
        timestamps_file = self.config.mps_path / "slam" / "closed_loop_trajectory.csv"
        if not timestamps_file.exists():
            raise FileNotFoundError(f"Aria timestamps file not found: {timestamps_file}")
        return pd.read_csv(timestamps_file, usecols=["tracking_timestamp_us", "utc_timestamp_ns"])

    def _synchronize_timestamps(self, aria_timestamps: pd.DataFrame, orbecc_timestamps: pd.DataFrame) -> pd.DataFrame:
        """
        Synchronize timestamps between Aria and Orbecc devices.

        Args:
            aria_timestamps: DataFrame containing Aria device timestamps
            orbecc_timestamps: DataFrame containing Orbecc device timestamps

        Returns:
            DataFrame with synchronized timestamps
        """
        # Calculate median timestamps for Orbecc frames
        timestamp_cols = [col for col in orbecc_timestamps.columns if col != "frame_number"]
        orbecc_median_timestamps = orbecc_timestamps[timestamp_cols].median(axis=1)

        # Initialize result with frame numbers
        result = pd.DataFrame()

        frame_nums = []
        tracking_timestamps = []
        for frame_num, orbecc_median_timestamp in zip(orbecc_timestamps["frame_number"], orbecc_median_timestamps):
            closest_idx = (aria_timestamps["utc_timestamp_ns"] - orbecc_median_timestamp).abs().idxmin()
            diff_ns = np.abs((aria_timestamps["utc_timestamp_ns"][closest_idx] - orbecc_median_timestamp))

            if diff_ns > 1e7:
                print(f"Skipping {frame_num}...Difference in nanoseconds: {diff_ns}")
                continue
            
            frame_nums.append(frame_num - 15)
            tracking_timestamps.append(aria_timestamps.loc[closest_idx, "tracking_timestamp_us"])

        result["frame_number"] = frame_nums
        result["tracking_timestamp_us"] = tracking_timestamps
        return result

    def _extract_frames(self, timestamps: pd.DataFrame, output_dirs: Dict[str, Path]) -> None:
        """
        Extract and process frames using the provided timestamps.

        Args:
            timestamps: DataFrame containing synchronized timestamps
            output_dirs: Dictionary mapping camera labels to output directories
        """
        options = self.provider.get_default_deliver_queued_options()
        rgb_stream_ids = options.get_stream_ids(RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS)
        options.set_subsample_rate(rgb_stream_ids[0], 1)

        # Initialize calibrations
        src_calib = self.device_calib.get_camera_calib(self.config.camera_label)
        dst_calib = calibration.get_linear_camera_calibration(
            self.config.output_image_size[0],
            self.config.output_image_size[1],
            self.config.focal_length,
            self.config.camera_label,
        )
        devignetting_mask = self.device_calib.load_devignetting_mask(self.config.camera_label)

        for _, row in tqdm(timestamps.iterrows(), total=len(timestamps)):
            self._process_single_frame(
                row["frame_number"],
                row["tracking_timestamp_us"],
                rgb_stream_ids[0],
                src_calib,
                dst_calib,
                devignetting_mask,
                output_dirs[self.config.camera_label],
            )

    def _process_single_frame(
        self,
        frame_number: int,
        device_timestamp: int,
        stream_id: int,
        src_calib: calibration.CameraCalibration,
        dst_calib: calibration.CameraCalibration,
        devignetting_mask: np.ndarray,
        output_dir: Path,
    ) -> None:
        """
        Process a single frame with the given parameters.

        Args:
            frame_number: Frame number for the output filename
            device_timestamp: Device timestamp in microseconds
            stream_id: RGB stream ID
            src_calib: Source camera calibration
            dst_calib: Destination camera calibration
            devignetting_mask: Devignetting mask for image correction
            output_dir: Output directory for the processed frame
        """
        image_data = self.provider.get_image_data_by_time_ns(
            stream_id,
            int(device_timestamp * 1000),
            TimeDomain.DEVICE_TIME,
            TimeQueryOptions.CLOSEST,
        )

        if not image_data:
            print(f"Warning: No image data found for frame {frame_number}")
            return

        # [INFO] Get wrist and palm pose
        T_world_device = self.mps_data_provider.get_closed_loop_pose(
            int(device_timestamp * 1000), TimeQueryOptions.CLOSEST
        ).transform_world_device.to_matrix()
        wrist_and_palm_pose = self.mps_data_provider.get_wrist_and_palm_pose(
            int(device_timestamp * 1000), TimeQueryOptions.CLOSEST
        )

        # hand_information = {
        #     "left_wrist": wrist_and_palm_pose.left_hand.wrist_position_device,
        #     "left_palm": wrist_and_palm_pose.left_hand.palm_position_device,
        #     "right_wrist": wrist_and_palm_pose.right_hand.wrist_position_device,
        #     "right_palm": wrist_and_palm_pose.right_hand.palm_position_device,
        #     "left_wrist_normal": wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.wrist_normal_device,
        #     "left_palm_normal": wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.palm_normal_device,
        #     "right_wrist_normal": wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.wrist_normal_device,
        #     "right_palm_normal": wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.palm_normal_device,
        # }

        # Transform to world coordinate system
        # for key in hand_information.keys():
        #     if hand_information[key] is not None:
        #         hand_information[key] = (T_world_device @ np.r_[hand_information[key], [1]])[:3]

        raw_image = image_data[0].to_numpy_array()

        # cv2.circle(raw_image, pos.astype(np.int32), 10, (255, 0, 0), -1)

        processed_image = self._transform_image(
            raw_image,
            devignetting_mask,
            dst_calib,
            src_calib,
        )

        output_path = output_dir / f"color_{int(frame_number):06d}_camera07.jpg"
        Image.fromarray(processed_image).save(output_path)
        # np.save(f"data/recordings/20250206_Testing/Aria/export/hand/hand_{int(frame_number):06d}.npy", hand_information)

    def _transform_image(
        self,
        raw_image: np.ndarray,
        devignetting_mask: np.ndarray,
        dst_calib: calibration.CameraCalibration,
        src_calib: calibration.CameraCalibration,
    ) -> np.ndarray:
        """
        Apply transformation pipeline to the raw image.

        Args:
            raw_image: Raw input image
            devignetting_mask: Devignetting mask for image correction
            dst_calib: Destination camera calibration
            src_calib: Source camera calibration

        Returns:
            Transformed image array
        """
        # Apply devignetting correction
        corrected_image = calibration.devignetting(raw_image, devignetting_mask)

        # Apply distortion correction
        undistorted_image = calibration.distort_by_calibration(
            corrected_image, dst_calib, src_calib, InterpolationMethod.BILINEAR
        )

        # Rotate image
        return np.rot90(undistorted_image, k=3)


def main():
    """Main entry point for the frame extraction process."""
    RECORDING = "20250519_Testing"
    config = ProcessingConfig(
        # aria_name="b0bb3408-2f28-46d0-bf48-ad2892708a1c",
        aria_name="929c8945-8451-4134-9de7-e483f2aa8f54",
        recording_name=RECORDING,
        base_dir=Path("data/recordings") / RECORDING / "Aria",
        devignetting_mask_path=Path("data/aria_devignetting_masks"),
    )

    output_dirs = {
        "camera-rgb": config.base_dir / "export" / "color",
    }

    try:
        processor = AriaFrameProcessor(config)
        processor.process_frames(output_dirs)
    except Exception as e:
        print(f"Error processing frames: {str(e)}")
        raise


if __name__ == "__main__":
    main()
