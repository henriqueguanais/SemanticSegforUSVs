from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
COLOR = (255, 0, 0)
THICKNESS = 2

model = YOLO("peso.pt")
classNames = ['buoy', 'cruise-ship', 'ferry-boat', 'freight-boat', 'inflatable-boat', 'kayak', 'motorboat', 'rock', 'sailboat']

def object_detect(frame):
    '''
    Faz a deteccao da base e takeoff, utilzando modelo pre-treinado do yolov8n.
    Recebe como argumento o frame atual do video.
    '''
    frame_copy = np.copy(frame)
    results = model(frame_copy, conf=0.6, max_det=1)
    x1, y1, x2, y2, cls = 0, 0, 0, 0, 0
    for r in results:
        boxes = r.boxes
        if len(boxes) >= 1:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
                cls = int(box.cls[0])

    org = [x1, y1]
    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.putText(frame_copy, classNames[cls], org, FONT, FONTSCALE, COLOR, THICKNESS)

    return frame_copy, [x1, y1, x2, y2]

# img_test = cv2.imread("00014050L.jpg")
# frame, coords = object_detect(img_test)
# print(coords)

# plt.imshow(frame)
# plt.show()


