import cv2
from measure_distance import DistanceMeter
from gps_mark import GPSMarker
import matplotlib.pyplot as plt
from yolo_test import object_detect
import numpy as np

calibration_dir = "calibration.yaml"
calibration_file = cv2.FileStorage(calibration_dir, cv2.FILE_STORAGE_READ)
Kl = np.array(calibration_file.getNode('M1').mat())
T = np.array(calibration_file.getNode('T').mat())   
calibration_file.release()

focal_length = Kl[0, 0]
baseline = T[0, 0]/-1000

stereo_method_file = "stereo-method.yaml"
imgL = cv2.imread("/home/henrique/projects/python/SemanticSegforUSVs/00013814L.jpg")
imgR = cv2.imread("/home/henrique/projects/python/SemanticSegforUSVs/00013814R.jpg")

imgL_detected, coords = object_detect(imgL)

x1, y1, x2, y2 = coords
print(f"Coordenadas do objeto: {x1}, {y1}, {x2}, {y2}")
x, y, w, h = x1, y1, x2-x1, y2-y1
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

gps_path = 'gps.txt'
imu_path = 'imu.txt'
magnetic_declination = 4.41
center_x = x + h/2
center_y = y + w/2

# cria o objeto DistanceMeter, responsavel por fazer o mapa de disparidade e calcular a distancia
distance_meter = DistanceMeter(focal_length, baseline)
# carrega os parametros da camera estereo
distance_meter.load_stereo_params(stereo_method_file)
# calcula a disparidade e a distancia do objeto
distance_meter.disparity_compute(imgL, imgR, (x, y), (w, h))
# plota os resultados, com ou sem os pontos extremos (disparidade minima e maxima)


# cria o objeto GPSMarker, responsavel por calcular as coordenadas do objeto
gps_marker = GPSMarker(magnetic_declination)
# calcula o angulo do objeto em relacao ao centro da camera
gps_marker.angle_object(center_x, distance_meter.depth, imgL, focal_length)
# calcula as coordenadas do objeto
obj_latitude, obj_longitude = gps_marker.gps_mark(gps_path, imu_path, utm_zone=33)

print(f"Coordenadas do barco: {gps_marker.boat_coords[0]}, {gps_marker.boat_coords[1]}")
print(f"Coordenadas do objeto: {gps_marker.obj_coords[0]}, {gps_marker.obj_coords[1]}")