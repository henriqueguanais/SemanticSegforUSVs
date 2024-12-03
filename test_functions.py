import cv2
from measure_distance import DistanceMeter
from gps_mark import GPSMarker

x, y, w, h = 517, 320, 100, 200
stereo_method_file = "stereo-method.yaml"
imgL = cv2.imread("00022024L.jpg")
imgR = cv2.imread("00022024R.jpg")
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
focal_length = 957.795  
baseline = 0.3564      

# cria e o plota o mapa de disparidade, calculando a distancia ate determinado objeto
distance_meter = DistanceMeter(focal_length, baseline)
distance_meter.load_stereo_params(stereo_method_file)
distance_meter.disparity_compute(imgL, imgR, (x, y), (w, h))
distance_meter.plot_results(plot_extreme_points=False)

gps_path = 'MODD2_GPS_data/gps/kope75-00-00021500-00022160/gps/00022024.txt'
imu_path = 'MODD2_video_data_rectified/video_data/kope75-00-00021500-00022160/imu/00022024.txt'
magnetic_declination = 3.14
center_x = x + h/2
center_y = y + w/2

# descobre as coordenadas do objeto em relacao ao barco
gps_marker = GPSMarker(magnetic_declination)
gps_marker.angle_object(center_x, center_y, distance_meter.depth, imgL)
obj_latitude, obj_longitude = gps_marker.gps_mark(gps_path, imu_path)

print(f"Coordenadas do barco: {gps_marker.boat_coords[0]}, {gps_marker.boat_coords[1]}")
print(f"Coordenadas do objeto: {gps_marker.obj_coords[0]}, {gps_marker.obj_coords[1]}")
