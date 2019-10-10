import sys
import cv2
import numpy as np
from skimage.exposure import rescale_intensity
# https://docs.opencv.org/3.4/dc/da3/tutorial_copyMakeBorder.html


yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])

rgb_from_yuv = np.linalg.inv(yuv_from_rgb)

def yuv2rgb(yuv):
    return np.clip(np.dot(yuv,rgb_from_yuv),0,1)

def gaussian_matrix():
	A = np.zeros([5,5])
	W = np.array([0.05, 0.25, 0.4, 0.25, 0.05])
	A[2] = W
	A[0][2]=A[4][2] = 0.05
	A[1][2]=A[3][2] = 0.25
	return A

def padded_image(image):
	borderType = cv2.BORDER_CONSTANT
	top = bottom = left = right = 2
	padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType)

	return padded_image

def gaussian_blur(img):
	image = img.copy()
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
				output[y-2,x-2,i] = k/(kernel.sum())
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output*255).astype('uint8')
	return output

def pyrDown(img):
	image = img.copy()
	scale = 0.5
	width = int(image.shape[1]*scale)
	height = int(image.shape[0]*scale)
	dim = (width, height)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def pyrUp(img):
	image = img.copy()
	scale = 2
	width = int(image.shape[1]*scale)
	height = int(image.shape[0]*scale)
	dim = (width, height)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

def gaussian_pyramid(image,levels):
	assert np.power(2,levels)<=image.shape[0] and np.power(2,levels)<=image.shape[1]
	layer = image.copy()
	arr = [layer]
	for i in range(levels):
		blurred = gaussian_blur(layer)
		layer = cv2.pyrDown(blurred)
		arr.append(layer)
	return arr
	
def print_gaussian(gaussian_pyramid):
	for i in range(len(gaussian_pyramid)):
		cv2.imshow(str(i), gaussian_pyramid[i])
		cv2.waitKey(0)
		
def print_laplacian_pyramid(laplacian_pyramid):
	for i in range(len(laplacian_pyramid):
		cv2.imshow(str(i), laplacian_pyramid[i])
		cv2.waitKey(0)

def laplacian_pyramid(image,levels):
	assert np.power(2,levels)<=image.shape[0] and np.power(2,levels)<=image.shape[1]
	layer = image.copy()
	gp = gaussian_pyramid(layer,levels)

	lp = []
	for i in range(1, levels+1):
		expanded_image = cv2.pyrUp(gp[i])
		j = i-1
		laplacian = cv2.subtract(gp[j], expanded_image)
		lp.append(laplacian)
	lp.append(gp[-1])
	return lp,gp

def reconstructed(lp):
	levels = len(lp)
	expanded_image = cv2.pyrUp(lp[-1])	
	for i in range(levels-2,-1,-1):
		corrected_image = cv2.add(expanded_image,lp[i])
		expanded_image = cv2.pyrUp(corrected_image)
	return corrected_image

def blending_with_overlapping_regions(lp1, lp2):
	LS = []
	for la,lb in zip(lp1, lp2):
		rows, cols, dpt = la.shape
		ls = np.hstack((la[:,0:int(cols/2),:], lb[:, int(cols/2):,:]))
		LS.append(ls)
	image = reconstructed(LS)
	return image

def blending_with_Arbitrary_regions(image1_l, image2_l, mask_g):
	LS = []
	for la,lb,gp in zip(image1_l, image2_l, mask_g):
		rows, cols, dpt = la.shape
		ls = cv2.add(gp*la, (1-gp)*lb)
		LS.append(ls)
	image = reconstructed(LS)
	return image


# file1 = sys.argv[1]
# file2 = sys.argv[2]
# file3 = sys.argv[3]

# image1 = cv2.imread(file1)
# image2 = cv2.imread(file2)
# image3 = cv2.imread(file3)

# cv2.imshow("Original1", image1)
# cv2.imshow("Original2", image2)
# cv2.waitKey(0)

# a1,b1 = laplacian_pyramid(image1, 5)
# a2,b2 = laplacian_pyramid(image2, 5)
# for i in range(5):
# 	cv2.imshow(str(i), a1[i])
# 	cv2.imshow(str(i), a2[i])
# 	cv2.waitKey(0)

# b3 = gaussian_pyramid(image3, 5)

# reconstructed1 = reconstructed(a1)
# reconstructed2 = reconstructed(a2)
# cv2.imshow("reconstructed1", reconstructed1)
# cv2.imshow("reconstructed2", reconstructed2)
# cv2.waitKey(0)

# blended_image = blending_with_overlapping_regions(a1, a2)
# cv2.imshow("blended_image", blended_image)
# cv2.waitKey(0)




# a,b = laplacian_pyramid(image,3)

# for i in range(len(a)):
# 	print(b[i].shape)
# 	cv2.imshow("image_pyramid"+str(i),b[i])
# cv2.waitKey(0)

# for i in range(len(a)):
# 	print(a[i].shape)
# 	cv2.imshow("image_pyramid"+str(i),a[i])
# cv2.waitKey(0)

# c = reconstructed(a)
# cv2.imshow("reconstructed", c)
# cv2.waitKey(0)
