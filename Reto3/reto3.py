import random
import math
import time
import csv
import cv2
import numpy as np
import os

path = r"C:\Users\dmpie\OneDrive\Documentos\Python\VisionIA\Reto2\reto2.png"

image = cv2.imread(path)
"""cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

imageCopy = image.copy()

shape = imageCopy.shape
print(f"Shape: {shape}")

b, g, r = cv2.split(imageCopy)

zones = [
    (408, 431, 418, 441),
    (419, 442, 429, 452),
    (430, 453, 440, 463),
    (441, 464, 451, 474),
    (452, 475, 462, 485),
    (562, 585, 213, 235)
]

for i, (y1, y2, x1, x2) in enumerate(zones, 1):
    shortImg = imageCopy[y1:y2, x1:x2]
    meanB = np.mean(shortImg[:, :, 0])
    meanG = np.mean(shortImg[:, :, 1])
    meanR = np.mean(shortImg[:, :, 2])
    
    stdB = np.std(shortImg[:, :, 0])
    stdG = np.std(shortImg[:, :, 1])
    stdR = np.std(shortImg[:, :, 2])
    
    print(f"Zone {i}:")
    print(f"Mean: B: {meanB:.2f}, G: {meanG:.2f}, R: {meanR:.2f}")
    print(f"Standard Deviation: B: {stdB:.2f}, G: {stdG:.2f}, R: {stdR:.2f}\n")
    
    #cv2.imshow(f"Zone {i}", shortImg)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
