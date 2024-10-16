import torch
import cv2


model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/yolov5_custom/weights/best.pt')

img = cv2.imread('crew.jpg')
results = model(img)
results.show()
