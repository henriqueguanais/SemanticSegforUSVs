from gps_mark import GPSMarker
import glob
import numpy as np

magnetic_declination = 3.14

gps_marker = GPSMarker(magnetic_declination)

imu_data = sorted(glob.glob("MODD2_video_data_rectified/video_data/kope82-00-00012030-00012700/imu/*.txt"))

for imu_path in imu_data:
    imu = gps_marker.get_imu_values(imu_path)

    magnetic_heading = np.arctan2(imu[1], imu[0])
    magnetic_heading = np.degrees(magnetic_heading)
    if magnetic_heading < 0:
        magnetic_heading = magnetic_heading + 360
    print(magnetic_heading)

gps_txt_path = "MODD2_GPS_data/gps/kope82-00-00012030-00012700/gps/00012494.txt"
gps_coords = gps_marker.get_gps_coords(gps_txt_path)

gps_coords = gps_marker.gps_transform(gps_coords)
gps_marker.new_angle = 0
gps_marker.distance = 20
obj_latitude, obj_longitude = gps_marker.gps_mark(gps_txt_path, imu_path)

print(f"Coordenadas do barco: {gps_marker.boat_coords[0]}, {gps_marker.boat_coords[1]}")
print(f"Coordenadas do objeto: {gps_marker.obj_coords[0]}, {gps_marker.obj_coords[1]}")
