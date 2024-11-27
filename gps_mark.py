from pyproj import Proj, Transformer
import numpy as np
import cv2

def gps_transform(utm_coords):
    utm_proj = Proj(proj="utm", zone=33, ellps="WGS84", south=False)
    wgs84_proj = Proj(proj="latlong", datum="WGS84")

    transformer = Transformer.from_proj(utm_proj, wgs84_proj)

    longitude, latitude = transformer.transform(utm_coords[0], utm_coords[1])
    # print(latitude, longitude)
    return [latitude, longitude]


def get_gps_coords(gps_txt_path):
    with open(gps_txt_path, 'r') as f:
        data = f.readlines()
        x = float(data[0][0:-2])
        y = float(data[1][0:-2])
    
    return [x, y]


def get_imu_values(imu_txt_path):
    with open(imu_txt_path, 'r') as f:
        data = f.readlines()
        x = float(data[0][0:-2])
        y = float(data[1][0:-2])
        z = float(data[2][0:-2])
    
    return [x, y, z]


def angle_object(center_ox, center_oy, distance, img):
    width = img.shape[1]
    height = img.shape[0]

    distance_o2c = np.sqrt((center_ox - height/2)**2 + (center_oy - width/2)**2)
    angle_o2c = np.arctan(distance_o2c/distance)

    if center_oy < width/2:
        new_angle = -angle_o2c
    elif center_oy > width/2:
        new_angle = angle_o2c
    else:
        new_angle = 0
    
    return new_angle
    

def gps_mark(gps_txt_path, imu_txt_path, distance, magnetic_declination, angle_o):
    gps_coords = get_gps_coords(gps_txt_path)
    gps_coords = gps_transform(gps_coords)
    imu_values = get_imu_values(imu_txt_path)

    magnetic_heading = np.arctan2(imu_values[1], imu_values[0])
    magnetic_heading = np.degrees(magnetic_heading)
    if magnetic_heading < 0:
        magnetic_heading = magnetic_heading + 360
    true_heading = magnetic_heading + magnetic_declination + angle_o
    # print(true_heading)

    theta = np.deg2rad(true_heading)
    x_ = distance * np.cos(theta)
    y_ = distance * np.sin(theta)
    # print(x_, y_)

    latitude_ = y_ / 118320
    longitude_ = x_ / (118320 * np.cos(np.deg2rad(gps_coords[1])))
    
    new_longitude = gps_coords[1] + longitude_
    new_latitude = gps_coords[0] + latitude_
    old_latitude = gps_coords[0]
    old_longitude = gps_coords[1]
    
    return new_latitude, new_longitude, old_latitude, old_longitude
    

# gps_path = 'MODD2_GPS_data/gps/kope75-00-00021500-00022160/gps/00022024.txt'
# imu_path = 'MODD2_video_data_rectified/video_data/kope75-00-00021500-00022160/imu/00022024.txt'
# magnetic_declination = 3.14

# img_path = '00022024L.jpg'
# img = cv2.imread(img_path)

# angle_o = angle_object(403, 571, 16, img)


# latitude, longitude = gps_mark(gps_path, imu_path, 16, magnetic_declination, angle_o)
# print(latitude, longitude)



    

