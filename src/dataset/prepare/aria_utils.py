from typing import Optional
import numpy as np
from projectaria_tools.core.stream_id import StreamId

def get_T_device_sensor(device_calibration, key: str):
    return device_calibration.get_transform_device_sensor("camera-rgb")

# Helper functions for reprojection and plotting
def get_point_reprojection(
    point_position_device: np.array, device_calibration, key: str
) -> Optional[np.array]:
    point_position_camera = get_T_device_sensor(device_calibration, key).inverse() @ point_position_device
    point_position_pixel = device_calibration.get_camera_calib("camera-rgb").project(point_position_camera)
    return point_position_pixel


def get_wrist_and_palm_pixels(wrist_and_palm_pose, device_calibration, key: str) -> np.array:
    NORMAL_VIS_LEN = 0.05
    left_wrist = get_point_reprojection(
        wrist_and_palm_pose.left_hand.wrist_position_device, device_calibration, key
    )
    left_palm = get_point_reprojection(
        wrist_and_palm_pose.left_hand.palm_position_device, device_calibration, key
    )
    right_wrist = get_point_reprojection(
        wrist_and_palm_pose.right_hand.wrist_position_device, device_calibration, key
    )
    right_palm = get_point_reprojection(
        wrist_and_palm_pose.right_hand.palm_position_device, device_calibration, key
    )
    left_wrist_normal_tip = None
    left_palm_normal_tip = None
    right_wrist_normal_tip = None
    right_palm_normal_tip = None
    left_normals = wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device
    if left_normals is not None:
        left_wrist_normal_tip = get_point_reprojection(
            wrist_and_palm_pose.left_hand.wrist_position_device
            + wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.wrist_normal_device
            * NORMAL_VIS_LEN,
            device_calibration,
            key,
        )
        left_palm_normal_tip = get_point_reprojection(
            wrist_and_palm_pose.left_hand.palm_position_device
            + wrist_and_palm_pose.left_hand.wrist_and_palm_normal_device.palm_normal_device
            * NORMAL_VIS_LEN,
            device_calibration,
            key,
        )
    right_normals = wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device
    if right_normals is not None:
        right_wrist_normal_tip = get_point_reprojection(
            wrist_and_palm_pose.right_hand.wrist_position_device
            + wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.wrist_normal_device
            * NORMAL_VIS_LEN,
            device_calibration,
            key,
        )
        right_palm_normal_tip = get_point_reprojection(
            wrist_and_palm_pose.right_hand.palm_position_device
            + wrist_and_palm_pose.right_hand.wrist_and_palm_normal_device.palm_normal_device
            * NORMAL_VIS_LEN,
            device_calibration,
            key,
        )
    return (
        left_wrist,
        left_palm,
        right_wrist,
        right_palm,
        left_wrist_normal_tip,
        left_palm_normal_tip,
        right_wrist_normal_tip,
        right_palm_normal_tip,
    )


def plot_wrists_and_palms(
    plt,
    left_wrist,
    left_palm,
    right_wrist,
    right_palm,
    left_wrist_normal_tip,
    left_palm_normal_tip,
    right_wrist_normal_tip,
    right_palm_normal_tip,
):
    def plot_point(point, color):
        plt.plot(*point, ".", c=color, mew=1, ms=20)

    def plot_arrow(point, vector, color):
        plt.arrow(*point, *vector, color=color)

    if left_wrist is not None:
        plot_point(left_wrist, "blue")
    if left_palm is not None:
        plot_point(left_palm, "blue")
    if right_wrist is not None:
        plot_point(right_wrist, "red")
    if right_palm is not None:
        plot_point(right_palm, "red")
    if left_wrist_normal_tip is not None:
        plot_arrow(left_wrist, left_wrist_normal_tip - left_wrist, "blue")
    if left_palm_normal_tip is not None:
        plot_arrow(left_palm, left_palm_normal_tip - left_palm, "blue")
    if right_wrist_normal_tip is not None:
        plot_arrow(right_wrist, right_wrist_normal_tip - right_wrist, "red")
    if right_palm_normal_tip is not None:
        plot_arrow(right_palm, right_palm_normal_tip - right_palm, "red")