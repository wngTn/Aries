import open3d as o3d
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence
import numpy as np

def load_point_cloud(global_points_path, inverse_distance_std_threshold=0.001, distance_std_threshold=0.1):
    points = mps.read_global_point_cloud(global_points_path)
    filtered_points = filter_points_from_confidence(points, inverse_distance_std_threshold, distance_std_threshold)
    world_positions = []
    for point in filtered_points:
        world_positions.append(point.position_world)

    world_positions = np.array(world_positions)

    return world_positions

from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId

vrsfile = "/Users/tonywang/Documents/hiwi/Aries/Code/Aries/data/recordings/20241023_Christian/raw/aria/mps_Christian_IFL_2/Christian_IFL_2.vrs"
provider = data_provider.create_vrs_data_provider(vrsfile)
assert provider is not None, "Cannot open file"

for stream_id in provider.get_all_streams():
  t_first = provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
  t_last = provider.get_last_time_ns(stream_id, TimeDomain.DEVICE_TIME)
  query_timestamp = (t_first + t_last) // 2 # example query timestamp
  sensor_data = provider.get_sensor_data_by_time_ns(stream_id, query_timestamp, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST)
