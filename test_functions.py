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
imgL = cv2.imread("00014050L.jpg")
imgR = cv2.imread("00014050R.jpg")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
plt.title("Left Image", fontsize=20)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
plt.title("Right Image", fontsize=20)
plt.axis('off')
plt.show()

imgL_detected, coords = object_detect(imgL)

plt.imshow(cv2.cvtColor(imgL_detected, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

x1, y1, x2, y2 = coords
print(f"Coordenadas do objeto: {x1}, {y1}, {x2}, {y2}")
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

gps_path = 'gps.txt'
imu_path = 'imu.txt'
magnetic_declination = 4.41
center_x = x1 + (x2 - x1)/2
center_y = y1 + (y2 - y1)/2

# cria o objeto DistanceMeter, responsavel por fazer o mapa de disparidade e calcular a distancia
distance_meter = DistanceMeter(focal_length, baseline)
# carrega os parametros da camera estereo
distance_meter.load_stereo_params(stereo_method_file)
# calcula a disparidade e a distancia do objeto
distance_meter.disparity_compute(imgL, imgR, (x1, y1), (x2, y2))
# plota os resultados, com ou sem os pontos extremos (disparidade minima e maxima)
distance_meter.plot_results(plot_extreme_points=False)
plt.imshow(distance_meter.disparity_map, cmap='gray')
plt.axis('off')
plt.show()
# cria o objeto GPSMarker, responsavel por calcular as coordenadas do objeto
gps_marker = GPSMarker(magnetic_declination)
# calcula o angulo do objeto em relacao ao centro da camera
gps_marker.angle_object(center_x, distance_meter.depth, imgL, focal_length)
# calcula as coordenadas do objeto
obj_latitude, obj_longitude = gps_marker.gps_mark(gps_path, imu_path, utm_zone=33)

print(f"Coordenadas do barco: {gps_marker.boat_coords[0]}, {gps_marker.boat_coords[1]}")
print(f"Coordenadas do objeto: {gps_marker.obj_coords[0]}, {gps_marker.obj_coords[1]}")