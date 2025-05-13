from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

vrsfile = "data/recordings/20241023_Christian/Aria/Christian_IFL_Last_2/Christian_IFL_Last_2.vrs"

print(f"Creating data provider from {vrsfile}")
provider = data_provider.create_vrs_data_provider(vrsfile)
if not provider:
    print("Invalid vrs data provider")
    
options = (
    provider.get_default_deliver_queued_options()
)  # default options activates all streams

options.set_truncate_first_device_time_ns(
    int(1e8)
)  # 0.1 secs after vrs first timestamp
options.set_truncate_last_device_time_ns(int(1e9))  # 1 sec before vrs last timestamp

# deactivate all sensors
options.deactivate_stream_all()
# activate only a subset of sensors
rgb_stream_ids = options.get_stream_ids(RecordableTypeId.RGB_CAMERA_RECORDABLE_CLASS)
# slam_stream_ids = options.get_stream_ids(RecordableTypeId.SLAM_CAMERA_DATA)
# imu_stream_ids = options.get_stream_ids(RecordableTypeId.SLAM_IMU_DATA)

# for stream_id in slam_stream_ids:
#     options.activate_stream(stream_id)  # activate slam cameras
#     options.set_subsample_rate(stream_id, 1)  # sample every data for each slam camera

# for stream_id in imu_stream_ids:
#     options.activate_stream(stream_id)  # activate imus
#     options.set_subsample_rate(stream_id, 10)  # sample every 10th data for each imu
    
for stream_id in rgb_stream_ids:
    options.activate_stream(stream_id)  # activate rgb cameras
    options.set_subsample_rate(stream_id, 1)  # sample every
    
    
provider = data_provider.create_vrs_data_provider(vrsfile)
options.set_subsample_rate(stream_id, 1)  # sample every data for camera
iterator = provider.deliver_queued_sensor_data(options)
for sensor_data in iterator:
    label = provider.get_label_from_stream_id(sensor_data.stream_id())
    sensor_type = sensor_data.sensor_data_type()
    device_timestamp = sensor_data.get_time_ns(TimeDomain.DEVICE_TIME)
    micro_sec = int(device_timestamp / 1000)
    
    # Sensor data obtained by timestamp (nanoseconds) 
    time_domain = TimeDomain.DEVICE_TIME  # query data based on DEVICE_TIME
    time_query_option = TimeQueryOptions.CLOSEST # get data whose time [in TimeDomain] is CLOSEST to query time
    image_data = provider.get_image_data_by_time_ns(stream_id, device_timestamp, time_domain, time_query_option)