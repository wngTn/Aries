
import open3d.visualization.gui as gui
import numpy as np
import open3d as o3d

def visualize_point_clouds(geometry_dict):
    app = gui.Application.instance
    app.initialize()
    vis = o3d.visualization.O3DVisualizer("Open3D", 1024, 768)
    vis.show_settings = True

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry("origin", mesh_frame)

    for name, geometry in geometry_dict.items():
        vis.add_geometry(name, geometry)

    app.add_window(vis)
    app.run()

def get_camera_poses(cam_infos, size=0.5):
    """
    Render camera poses as coordinate frames.
    This function takes a dictionary of camera information and generates
    coordinate frames for each camera based on their extrinsic parameters.
    The coordinate frames are transformed according to the provided extrinsics
    and returned in a dictionary.
    Args:
        cam_infos (dict): A dictionary where keys are camera names and values
                          are dictionaries containing camera information. Each
                          camera information dictionary must have an 'extrinsics'
                          key with a 4x4 transformation matrix as its value.
        size (float, optional): The size of the coordinate frames. Default is 0.5.
    Returns:
        dict: A dictionary where keys are camera names appended with '_pose' and
              values are the transformed coordinate frames as Open3D TriangleMesh
              objects.
    """
    
    camera_coordinate_frames = {}
    for cam_name, cam_info in cam_infos.items():
        cam2world = cam_info['extrinsics']
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
        mesh_frame.transform(cam2world)
        camera_coordinate_frames[f"{cam_name}_pose"] = mesh_frame
    
    return camera_coordinate_frames