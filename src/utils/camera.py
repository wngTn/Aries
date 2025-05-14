from pathlib import Path
from scipy.spatial.transform import Rotation
import json
import numpy as np


def load_rotation_matrix(rot: dict) -> np.ndarray:
    """
    Convert a quaternion rotation dictionary to a 3x3 rotation matrix.

    Parameters
    ----------
    rot : dict
        Dictionary containing quaternion rotation parameters with keys 'x', 'y', 'z', 'w'
        representing the quaternion components.

    Returns
    -------
    np.ndarray
        A 3x3 rotation matrix representing the same rotation as the input quaternion.

    Examples
    --------
    >>> rot = {'x': 0, 'y': 0, 'z': 0, 'w': 1}  # Identity quaternion
    >>> load_rotation_matrix(rot)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    """
    return Rotation.from_quat([rot['x'], rot['y'], rot['z'], rot['w']]).as_matrix()


def load_transform_matrix(trans: dict, rot: dict) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from translation and rotation components.

    Parameters
    ----------
    trans : dict
        Dictionary containing translation parameters with keys 'm00', 'm10', 'm20'
        representing x, y, z translations respectively.
    rot : dict
        Dictionary containing quaternion rotation parameters with keys 'x', 'y', 'z', 'w'.

    Returns
    -------
    np.ndarray
        A 4x4 homogeneous transformation matrix combining the rotation and translation.
        The matrix has the form:
        [R R R Tx]
        [R R R Ty]
        [R R R Tz]
        [0 0 0  1]
        where R represents rotation components and T represents translation components.
    """
    transform = np.zeros((4, 4), dtype=np.float32)
    transform[:3, :3] = load_rotation_matrix(rot)
    transform[:, 3] = [trans['m00'], trans['m10'], trans['m20'], 1]
    return transform


def extract_intrinsics_matrix(intrinsics_json: dict) -> np.ndarray:
    """
    Convert a camera intrinsics dictionary to a 3x3 camera intrinsics matrix.

    Parameters
    ----------
    intrinsics_json : dict
        Dictionary containing camera intrinsics parameters with keys 'm00' through 'm22'
        representing elements of the 3x3 intrinsics matrix.

    Returns
    -------
    np.ndarray
        A 3x3 camera intrinsics matrix containing focal lengths and principal point offsets:
        [[fx  s  cx]
         [0   fy cy]
         [0   0   1]]
    """
    return np.asarray([[intrinsics_json['m00'], intrinsics_json['m10'], intrinsics_json['m20']],
                    [intrinsics_json['m01'], intrinsics_json['m11'], intrinsics_json['m21']],
                    [intrinsics_json['m02'], intrinsics_json['m12'], intrinsics_json['m22']]])

def rotation_to_homogenous(vec):
    """
    Convert a rotation vector to a 4x4 homogeneous transformation matrix.

    Parameters
    ----------
    vec : np.ndarray
        A 3D rotation vector specifying rotation axis and magnitude.

    Returns
    -------
    np.ndarray
        A 4x4 homogeneous transformation matrix representing the rotation:
        [[R R R 0]
         [R R R 0]
         [R R R 0]
         [0 0 0 1]]
        where R represents the rotation components.
    """
    rot_mat = Rotation.from_rotvec(vec)
    swap = np.identity(4)
    swap = np.zeros((4, 4))
    swap[:3, :3] = rot_mat.as_matrix()
    swap[3, 3] = 1
    return swap

def convert_extrinsics(extrinsics):
    # Extract the rotation matrix (3x3) and translation vector (3x1)
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]

    # Define the reflection matrix for flipping the Z-axis
    reflection_matrix = np.diag([1, 1, -1])

    # Apply the reflection to the rotation matrix and translation vector
    R_new = np.dot(R, reflection_matrix)
    t_new = np.dot(reflection_matrix, t)

    # Construct the new extrinsic matrix
    extrinsics_new = np.hstack((R_new, t_new.reshape(-1, 1)))
    extrinsics_new = np.vstack((extrinsics_new, [0, 0, 0, 1]))

    return extrinsics_new

def load_cam_infos(take_path: Path) -> dict:
    """
    Load and process camera calibration information from JSON files in a directory.

    Parameters
    ----------
    take_path : Path
        Path to the directory containing a 'calibration' subdirectory with camera
        calibration JSON files named 'camera*.json'.

    Returns
    -------
    dict
        Dictionary containing camera parameters for each camera, with keys 'camera0X' where
        X is the camera number. Each camera's dictionary contains:
        - K: 3x3 camera intrinsics matrix
        - T_world_camera: 4x4 camera extrinsics matrix, p_world = T_world_camera @ p_camera
        - fov_x: horizontal field of view
        - fov_y: vertical field of view
        - c_x: principal point x-coordinate
        - c_y: principal point y-coordinate
        - width: image width in pixels
        - height: image height in pixels
        - radial_params: radial distortion parameters
        - tangential_params: tangential distortion parameters

    Notes
    -----
    The function applies several coordinate transformations including YZ flip and swap
    to align with a specific coordinate system convention.
    """
    camera_parameters = {}
    camera_paths = sorted((take_path / "calibration").glob('camera*.json'))

    for camera_path in camera_paths:
        cam_idx = camera_path.stem
        with camera_path.open() as f:
            cam_info = json.load(f)['value0']

        intrinsics = extract_intrinsics_matrix(cam_info['color_parameters']['intrinsics_matrix'])
        intrinsics[2, 1] = 0
        intrinsics[2, 2] = 1
        extrinsics = load_transform_matrix(cam_info['camera_pose']['translation'], cam_info['camera_pose']['rotation'])

        # [INFO] Orbbec Cameras
        if 'color2depth_transform' in cam_info:
            color2depth_transform = load_transform_matrix(cam_info['color2depth_transform']['translation'],
                                                        cam_info['color2depth_transform']['rotation'])

            extrinsics = np.matmul(extrinsics, color2depth_transform)

            YZ_FLIP = rotation_to_homogenous(np.pi * np.array([1, 0, 0]))
            YZ_SWAP = rotation_to_homogenous(np.pi/2 * np.array([1, 0, 0]))

            extrinsics = YZ_SWAP @ extrinsics @ YZ_FLIP

        # [INFO] Transform the Marshall Cameras
        else:
            T_marshall_orbbec = np.eye(4)
            T_marshall_orbbec[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
            extrinsics = np.linalg.inv(extrinsics)
            extrinsics = T_marshall_orbbec @ extrinsics

        color_params = cam_info['color_parameters']
        radial_params = tuple(color_params['radial_distortion'].values())
        tangential_params = tuple(color_params['tangential_distortion'].values())

        camera_parameters[cam_idx] = {
            'K': intrinsics,
            'T_world_camera': extrinsics,
            'fov_x': color_params['fov_x'],
            'fov_y': color_params['fov_y'],
            'c_x': color_params['c_x'],
            'c_y': color_params['c_y'],
            'width': color_params['width'],
            'height': color_params['height'],
            'radial_params': radial_params,
            'tangential_params': tangential_params,
        }

    return camera_parameters

# Additional helper function to project 3D points to 2D using camera parameters
def project_to_2d(points_3d, K, T_camera_world):
    """
    Project a batch of 3D points to 2D image coordinates using camera parameters.

    Parameters
    ----------
    points_3d : np.ndarray
        Batch of 3D point coordinates in world space with shape (3, N).
    K : np.ndarray
        3x3 camera intrinsics matrix.
    T_camera_world : np.ndarray
        4x4 camera extrinsics matrix (p_camera = T_camera_world @ p_world).

    Returns
    -------
    np.ndarray
        Batch of 2D integer pixel coordinates with shape (2, N) of the projected points in the image plane.

    Notes
    -----
    The projection pipeline:
    1. Convert 3D points to homogeneous coordinates
    2. Transform to camera space using extrinsic matrix
    3. Project to image plane using intrinsic matrix
    4. Normalize by depth (z-coordinate)
    5. Convert to integer pixel coordinates
    """
    # Get number of points
    n_points = points_3d.shape[-1]

    # Convert the points to homogeneous coordinates (4, N)
    points_3d_hom = np.r_[points_3d, np.ones((1, n_points))]

    # Apply extrinsic matrix to all points (4, N)
    points_cam = T_camera_world @ points_3d_hom

    # Apply intrinsic matrix to all points (3, N)
    points_img_hom = K @ points_cam[:3]

    # Normalize by the third (z) coordinate to get image coordinates
    z_coords = points_img_hom[2]

    # Normalize x, y by z to get (2, N) image coordinates
    points_img = points_img_hom[:2, :] / z_coords

    return points_img.astype(np.int32)