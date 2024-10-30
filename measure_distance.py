import cv2
import numpy as np
import matplotlib.pyplot as plt

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
imgL = cv2.imread("00014521L.jpg", cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread("00014521R.jpg", cv2.IMREAD_GRAYSCALE)

# Calcular disparidade
disparity_map = stereo.compute(imgL, imgR).astype(np.float32)/16

disparityImg = cv2.normalize(src=disparity_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)

focal_length = 957.8  # em pixels
baseline = 0.356      # em metros

# Definir uma região específica (x, y, largura, altura) no mapa de disparidade
x, y, w, h = 120, 590, 70, 70  # Ajuste conforme a região desejada

# Extrair o valor médio de disparidade da região
region_disparity = disparity_map[y:y+h, x:x+w]
mean_disparity = np.mean(region_disparity)

# Calcular a profundidade usando a média da disparidade
if mean_disparity > 0:  # Evitar divisão por zero
    depth = (focal_length * baseline) / mean_disparity
    print(f"Distância estimada na região ({x}, {y}) é: {depth:.2f} metros")
else:
    print("Disparidade insuficiente para calcular a distância.")

disparity_max = np.max(disparity_map)
disparity_min = np.min(disparity_map[disparity_map > 0])  # ignora valores zero ou negativos

# Calcular profundidade mínima e máxima
if disparity_min > 0:  # Verifica se há valores válidos de disparidade
    depth_min = (focal_length * baseline) / disparity_max
    depth_max = (focal_length * baseline) / disparity_min
    print(f"Profundidade mínima: {depth_min:.2f} m")
    print(f"Profundidade máxima: {depth_max:.2f} m")
else:
    print("Erro: disparidade insuficiente para calcular profundidade.")
    
# cv2.imshow('LEFT Rectified', imgL)
# cv2.imshow('RIGHT Rectified', imgR)
# cv2.imshow("Disparity ColorMap", disparityImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

disparityImg_rgb = cv2.cvtColor(disparityImg, cv2.COLOR_BGR2RGB)
cv2.rectangle(imgL, (x, y), (x+w, y+h), (255, 255, 255), 2)
cv2.rectangle(imgR, (x, y), (x+w, y+h), (255, 255, 255), 2)
cv2.rectangle(disparityImg_rgb, (x, y), (x+w, y+h), (255, 255, 255), 2)

# Exibir as imagens com o Matplotlib
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(imgL, cmap='gray')
plt.title("Imagem Esquerda (L)")
plt.text(10, 1200, f"Distância estimada na região ({x}, {y}) é: {depth:.2f} metros", fontsize=12, color='blue')
plt.text(10, 1300, f"Profundidade mínima: {depth_min:.2f} m", fontsize=12, color='blue')
plt.text(10, 1400, f"Profundidade máxima: {depth_max:.2f} m", fontsize=12, color='blue')
plt.axis('off')

plt.subplot(132)
plt.imshow(imgR, cmap='gray')
plt.title("Imagem Direita (R)")
plt.axis('off')

plt.subplot(133)
plt.imshow(disparityImg_rgb)
plt.title("Mapa de Disparidade")
plt.axis('off')
plt.show()