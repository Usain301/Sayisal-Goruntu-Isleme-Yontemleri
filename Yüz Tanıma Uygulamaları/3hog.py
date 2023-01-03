# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 11:39:37 2023

@author: cvkme
"""

import cv2
from skimage.feature import hog
from skimage import exposure

image = cv2.imread("images/elon_musk.jpg")

_, hogImage = hog(image, visualize=True)
rescaledImage = exposure.rescale_intensity(hogImage, in_range=(0,10))

cv2.imshow("HOG",hogImage)
cv2.imshow("rescaled Image HOG", rescaledImage)

