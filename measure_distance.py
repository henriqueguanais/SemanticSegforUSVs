import cv2
import numpy as np
import matplotlib.pyplot as plt
from gps_mark import gps_mark, angle_object

stereo_method_file = "stereo-method.yaml"

# Carregar o arquivo de parâmetros estéreo com o OpenCV
cv_file = cv2.FileStorage(stereo_method_file, cv2.FILE_STORAGE_READ)
if not cv_file.isOpened():
    print("Erro: Não foi possível abrir o arquivo de parâmetros estéreo.")
else:
    # Extrair os parâmetros do método estéreo
    PreFilterType = int(cv_file.getNode("PreFilterType").real())
    PreFilterSize = int(cv_file.getNode("PreFilterSize").real())
    PreFilterCap = int(cv_file.getNode("PreFilterCap").real())
    SADWindowSize = int(cv_file.getNode("SADWindowSize").real())
    MinDisparity = int(cv_file.getNode("MinDisparity").real())
    NumDisparities = int(cv_file.getNode("NumDisparities").real())
    TextureThreshold = int(cv_file.getNode("TextureThreshold").real())
    UniquenessRatio = int(cv_file.getNode("UniquenessRatio").real())
    SpeckleWindowSize = int(cv_file.getNode("SpeckleWindowSize").real())
    SpeckleRange = int(cv_file.getNode("SpeckleRange").real())
    Disp12MaxDiff = int(cv_file.getNode("Disp12MaxDiff").real())
    cv_file.release()

# Configurar StereoBM usando parâmetros carregados
stereo = cv2.StereoBM_create(
    numDisparities=NumDisparities,
    blockSize=SADWindowSize
)

stereo.setPreFilterType(PreFilterType)
stereo.setPreFilterSize(PreFilterSize)
stereo.setPreFilterCap(PreFilterCap)
stereo.setTextureThreshold(TextureThreshold)
stereo.setUniquenessRatio(UniquenessRatio)
stereo.setSpeckleWindowSize(SpeckleWindowSize)
stereo.setSpeckleRange(SpeckleRange)
stereo.setDisp12MaxDiff(Disp12MaxDiff)

# Leitura e pré-processamento das imagens (certifique-se de ajustar para o tamanho correto)
imgL = cv2.imread("00022024L.jpg")
imgR = cv2.imread("00022024R.jpg")
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Calcular disparidade

disparity_map = stereo.compute(imgL, imgR).astype(np.float32)/16
disparity_normalize = cv2.normalize(src=disparity_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparity_normalize = cv2.applyColorMap(disparity_normalize, cv2.COLORMAP_JET)

focal_length = 957.795  # em pixels
baseline = 0.3564      # em metros

# Definir uma região específica (x, y, largura, altura) no mapa de disparidade
x, y, w, h = 517, 320, 200, 100  # Ajuste conforme a região desejada

# Extrair o valor médio de disparidade da região
region_disparity = disparity_map[x:x+w, y:y+h]
mean_disparity = np.mean(region_disparity)

# Calcular a profundidade usando a média da disparidade
if mean_disparity > 0:  # Evitar divisão por zero
    depth = (focal_length * baseline) / mean_disparity
    print(f"Distância estimada na região ({x}, {y}) é: {depth:.2f} metros")
else:
    print("Disparidade insuficiente para calcular a distância.")

disparity_max = np.max(disparity_map)
disparity_min = np.min(disparity_map[disparity_map > 0])  # ignora valores zero ou negativos'

# Calcular profundidade mínima e máxima
if disparity_min > 0:  # Verifica se há valores válidos de disparidade
    depth_min = (focal_length * baseline) / disparity_max
    depth_max = (focal_length * baseline) / disparity_min
    print(f"Profundidade mínima: {depth_min:.2f} m")
    print(f"Profundidade máxima: {depth_max:.2f} m")
else:
    print("Erro: disparidade insuficiente para calcular profundidade.")

min_disparity_mask = (disparity_map == disparity_min)
max_disparity_mask = (disparity_map == disparity_max)

cv2.rectangle(imgL, (x, y), (x+h, y+w), (0, 0, 255), 3)
cv2.rectangle(imgR, (x, y), (x+h, y+w), (0, 0, 255), 3)
cv2.rectangle(disparity_normalize, (x, y), (x+h, y+w), (255, 255, 255), 3)
for y in range(disparity_map.shape[0]):
    for x in range(disparity_map.shape[1]):
        if min_disparity_mask[y, x]:
            cv2.circle(disparity_normalize, (x, y), radius=5, color=(0, 0, 0), thickness=-1)

for y in range(disparity_map.shape[0]):
    for x in range(disparity_map.shape[1]):
        if max_disparity_mask[y, x]:
            cv2.circle(disparity_normalize, (x, y), radius=5, color=(255, 255, 255), thickness=-1)

x, y, w, h = 517, 320, 100, 200
gps_path = 'MODD2_GPS_data/gps/kope75-00-00021500-00022160/gps/00022024.txt'
imu_path = 'MODD2_video_data_rectified/video_data/kope75-00-00021500-00022160/imu/00022024.txt'
magnetic_declination = 3.14
center_x = x + h/2
center_y = y + w/2
print(center_x, center_y)
angle_o = angle_object(center_x, center_y, depth, imgL)

latitude, longitude, old_latitude, old_longitude = gps_mark(gps_path, imu_path, depth, magnetic_declination, angle_o)
print(old_latitude, old_longitude)
print(latitude, longitude)

# Exibir as imagens com o Matplotlib
plt.figure(figsize=(16, 10))
plt.subplot(131)
plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
plt.title("Imagem Esquerda (L)")
# plt.text(10, 1200, f"Distância estimada na região é: {depth:.2f} metros", fontsize=12, color='blue')
# plt.text(10, 1300, f"Profundidade mínima: {depth_min:.2f} m", fontsize=12, color='blue')
# plt.text(10, 1400, f"Profundidade máxima: {depth_max:.2f} m", fontsize=12, color='blue')
plt.text(1, 1000, f"Centro do objeto: {center_x, center_y}", fontsize=12, color='blue')
plt.axis('off')

# plt.subplot(132)
# plt.imshow(imgR, cmap='gray')
# plt.title("Imagem Direita (R)")
# plt.axis('off')

gps_img = cv2.imread("gps2.png")
# plt.subplot(133)
plt.subplot(132)
plt.imshow((cv2.cvtColor(disparity_normalize, cv2.COLOR_BGR2RGB)))
plt.title("Mapa de Disparidade")
plt.text(10, 1000, f"Distância estimada na região é: {depth:.2f} metros", fontsize=12, color='blue')
plt.text(10, 1100, f"Profundidade mínima: {depth_min:.2f} m", fontsize=12, color='blue')
plt.text(10, 1200, f"Profundidade máxima: {depth_max:.2f} m", fontsize=12, color='blue')
plt.axis('off')

plt.subplot(133)
plt.imshow((cv2.cvtColor(gps_img, cv2.COLOR_BGR2RGB)))
plt.title("GPS mark")
plt.text(1, 400, f"COORDENADAS ATUAIS - {old_latitude}, {old_longitude}", fontsize=12, color='blue')
plt.text(1, 450, f"COORDENADAS OBJETO - {latitude}, {longitude}", fontsize=12, color='blue')
plt.axis('off')
plt.show()


