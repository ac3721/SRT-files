import matplotlib.pyplot as plt
import cv2
import numpy as np

from PIL import Image
import numpy as np
import pytesseract
import cv2
import os
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def find_text(img, debug = False):
    text = None
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([145, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Extract blue text from image
    blue_text_img = cv2.bitwise_and(img, img, mask=mask)
    if debug:
        plt.imshow(cv2.cvtColor(blue_text_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        print(cv2.countNonZero(mask))
    if cv2.countNonZero(mask) > 2000:
        text = blue_text_img
    return text

img = cv2.imread('image.png')
if img is None:
    raise ValueError("Could not load image")  
text = find_text(img, True)
