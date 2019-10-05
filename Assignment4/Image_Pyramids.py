import cv2
import numpy as np
import sys

filename = sys.argv[1]
image = cv2.imread(filename)
cv2.imshow("original", image)

# Create gaussian pyramid
layer = image.copy()
gaussian_pyramid = [layer]
for i in range(0,5):
	layer = cv2.pyrDown(layer)
	gaussian_pyramid.append(layer)

# print gaussian pyramid
for i in range(0, 5):
 	cv2.imshow(str(i), gaussian_pyramid[i])

# Create laplacian pyramid
laplacian_pyramid = []
for i in range(1, 5):
	expanded_image = cv2.pyrUp(gaussian_pyramid[i])
	j = i-1
	laplacian = cv2.subtract(gaussian_pyramid[j], expanded_image)
	laplacian_pyramid.append(laplacian)

laplacian_pyramid.append(gaussian_pyramid[4])

# Print laplacian pyramid
for i in range(0, 5):
 	cv2.imshow(str(i), laplacian_pyramid[i])

# To create the original image
expanded_image = cv2.pyrUp(laplacian_pyramid[4])

for i in range(3, -1, -1):
	corrected_image = cv2.add(expanded_image, laplacian_pyramid[i])
	expanded_image = cv2.pyrUp(corrected_image)

cv2.imshow("reconstructed", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
