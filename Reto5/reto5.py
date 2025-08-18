import random
import cv2
import numpy as np

path = r"C:\Users\dmpie\OneDrive\Documentos\Python\VisionIA\Reto5\reto5.png"

image = cv2.imread(path)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def padImage(img, pad_h, pad_w, mode='edge'):
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)

def convolution(image, kernel):
    kh, kw = kernel.shape
    ih, iw = image.shape
    pad_h, pad_w = kh // 2, kw // 2
    img_p = padImage(image, pad_h, pad_w, mode='edge').astype(np.float32)
    out = np.zeros((ih, iw), dtype=np.float32)

    k = np.flipud(np.fliplr(kernel)).astype(np.float32)

    for i in range(ih):
        for j in range(iw):
            region = img_p[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k)
    return np.clip(out, 0, 255).astype(np.uint8)

def medianFilter(image, ksize=3):
    pad = ksize // 2
    img_p = padImage(image, pad, pad, mode='edge')
    out = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = img_p[i:i+ksize, j:j+ksize]
            out[i, j] = np.median(region)
    return out

def addNoisy(imageGray):
    imgCopy = imageGray.copy()
    row , col = imageGray.shape
    
    pixels = random.randint(10000, 50000)
    for i in range(pixels):

        y_coord=random.randint(0, row - 1)

        x_coord=random.randint(0, col - 1)

        imgCopy[y_coord][x_coord] = 255

    for i in range(pixels):

        y_coord=random.randint(0, row - 1)

        x_coord=random.randint(0, col - 1)

        imgCopy[y_coord][x_coord] = 0

    return imgCopy

meanKernel = np.ones((3, 3), np.float32) / 9
gaussKernel = np.array([[1, 2, 1],
                         [2, 4, 2],
                         [1, 2, 1]], np.float32) / 16


imageCopy = addNoisy(imageGray)

meanFiltered = convolution(imageCopy, meanKernel)
gaussFiltered = convolution(imageCopy, gaussKernel)
medianFiltered = medianFilter(imageCopy, ksize=3)
medianCv2 = cv2.medianBlur(imageCopy, 3)

cv2.imshow("Original", imageGray)
cv2.imshow("Noisy", imageCopy)
cv2.imshow("Mean", meanFiltered)
cv2.imshow("Gauss", gaussFiltered)
cv2.imshow("Median", medianFiltered)
cv2.imshow("Filtro Mediana (cv2)", medianCv2)

cv2.waitKey(0)
cv2.destroyAllWindows()
