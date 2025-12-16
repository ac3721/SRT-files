import cv2
import numpy as np

color = np.uint8([[[45, 105, 160]]])

hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

print(hsv)
