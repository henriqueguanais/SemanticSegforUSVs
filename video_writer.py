import cv2 as cv
import os

# Pasta onde estão as imagens
image_folder = 'results3'
# Nome do arquivo de saída do vídeo
output_video = 'output_video2.avi'

# Pegar lista de imagens no diretório
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

# Ordenar as imagens, caso necessário
images.sort()  # Se você quiser garantir uma ordem específica

# Carregar a primeira imagem para obter as dimensões
frame = cv.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Definir codec e criar objeto VideoWriter
# cv.VideoWriter_fourcc(*'XVID') é o codec que define o formato de vídeo (pode ser alterado)
fourcc = cv.VideoWriter_fourcc(*'XVID')
video = cv.VideoWriter(output_video, fourcc, 10.0, (width, height))

# Loop sobre as imagens e adicioná-las ao vídeo
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv.imread(img_path)
    
    # Escrever o frame no vídeo
    video.write(frame)

# Libera o objeto VideoWriter
video.release()

print(f"Vídeo salvo como {output_video}")
