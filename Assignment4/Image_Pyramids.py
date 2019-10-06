import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
# https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html

def gaussian_matrix():
	A = np.zeros([5,5])
	W = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
	A[2] = W
	A[0][2]=A[4][2] = 0.05
	A[1][2]=A[3][2] = 0.25
	print( A )
	return A

def padded_image(image):
	borderType = cv2.BORDER_CONSTANT
	top = bottom = left = right = 2
	padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType)

	return padded_image

def gaussian_blur(image):
	height, width, ch = image.shape
	print(image.shape)
	image = padded_image(image)
	output = np.zeros((height, width, ch), dtype="float32")
	kernel = gaussian_matrix()
	for i in range(ch):
		for y in range(2, height+2):
			for x in range(2, width+2):
				roi = image[y-2:y+3, x-2:x+3, i]
				k = (roi*kernel).sum()
				output[y-2,x-2,i] = k
	output = rescale_intensity(output, in_range=(0, 255))
	blurred = (output*255).astype("uint8")
	return blurred

def pyrDown(image):
	scale = 0.5
	width = int(image.shape[1]*scale)
	height = int(image.shape[0]*scale)
	dim = (width, height)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def pyrUp(image):
	scale = 2
	width = int(image.shape[1]*scale)
	height = int(image.shape[0]*scale)
	dim = (width, height)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def gaussian_pyramid(image):
	layer = image.copy()
	arr = [layer]
	for i in range(5):
		blurred = gaussian_blur(layer)
		layer = pyrDown(blurred)
		arr.append(layer)
	
def print_gaussian(gaussian_pyramid):
	for i in range(5):
		cv2.imshow(str(i), gaussian_pyramid[i])
		cv2.waitKey(0)
	return gp

def laplacian_pyramid(image):
	layer = image.copy()
	gp = [layer]
	for i in range(5):
		blurred = gaussian_blur(layer)
		layer = pyrDown(layer)
		gp.append(layer)

	for i in range(5):
		cv2.imshow(str(i), gp[i])

	cv2.waitKey(0)
	lp = []
	for i in range(1, 5):
		expanded_image = pyrUp(gp[i])
		j = i-1
		laplacian = cv2.subtract(gp[j], expanded_image)
		lp.append(laplacian)
	lp.append(gp[4])
	return lp

def reconstructed(laplacian_pyramid):
	expanded_image = pyrUp(laplacian_pyramid[4])
	for i in range(3, -1, -1):
		corrected_image = cv2.add(expanded_image, laplacian_pyramid[i])
		expanded_image = pyrUp(corrected_image)
	return corrected_image

file = sys.argv[1]
image = cv2.imread(file)
cv2.imshow("Original", image)
cv2.waitKey(0)


a = laplacian_pyramid(image)
for i in range(5):
	cv2.imshow(str(i), a[i])
	cv2.waitKey(0)

b = reconstructed(a)
cv2.imshow("reconstructed", b)
cv2.waitKey(0)
