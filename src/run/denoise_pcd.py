import open3d as o3d
from tqdm import tqdm

from pathlib import Path

import rootutils

rootutils.setup_root(__file__, ".project-root", pythonpath=True, dotenv=True)

from lib.utils.point_cloud import denoise_point_cloud, voxelize_point_cloud

pcd_files = sorted(Path("data/recordings/20241213_Testing/Orbbec/pointclouds_fused").iterdir())

output_path = Path("data/recordings/20241213_Testing/Orbbec/pointclouds_denoised")
output_path.mkdir(exist_ok=True)

for pcd_file in tqdm(pcd_files):
    pcd = o3d.io.read_point_cloud(str(pcd_file))
    pcd_voxelized = voxelize_point_cloud(pcd, voxel_size=0.03)
    pcd_denoised = denoise_point_cloud(pcd_voxelized)[0]
    o3d.io.write_point_cloud(str(output_path / pcd_file.name), pcd_denoised)
    print(f"Saved denoised point cloud to {output_path / pcd_file.name}")

