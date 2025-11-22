import os
import cv2
from PIL import Image

folder = 'Test'
image_path = folder + '/one.png'

if not os.path.exists(image_path):
    print(f"Image file does not exist: {image_path}")
else:
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image, check file integrity or format.")

for image_path in os.listdir(folder):
    print(f"Image path type: {type(image_path)}, value: {image_path}")
    base_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # base_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow(base_image)
    # image = Image.fromarray(base_image)