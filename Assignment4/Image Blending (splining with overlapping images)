import cv2
import numpy as np
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)


row,col,ch = img1.shape
A = np.hstack((img1[:,0:int(col/2),:], img2[:, int(col/2):,:]))

cv2.imshow("without", A)

# Gaussian pyramid
layer1 = img1.copy()
layer2 = img2.copy()

gp1 = [layer1]
gp2 = [layer2]

for i in range(0,5):
	layer1 = cv2.pyrDown(layer1)
	gp1.append(layer1)
	layer2 = cv2.pyrDown(layer2)
	gp2.append(layer2)

# for i in range(0, 2):
# 	cv2.imshow(str(i), gp1[i])
# 	cv2.imshow(str(i), gp2[i])

# Laplacian pyramid
lp1 = []
lp2 = []
for i in range(1, 5):
	expanded_img1 = cv2.pyrUp(gp1[i])
	j1 = i-1
	laplacian1 = cv2.subtract(gp1[j1], expanded_img1)
	lp1.append(laplacian1)
	
	expanded_img2 = cv2.pyrUp(gp2[i])
	j2 = i-1
	laplacian2 = cv2.subtract(gp2[j2], expanded_img2)
	lp2.append(laplacian2)

lp1.append(gp1[4])
lp2.append(gp2[4])

# for i in range(0, 5):
# 	cv2.imshow(str(i), lp1[i])
# 	print( lp1[i].shape )

# cv2.waitKey(0)

# for i in range(0, 5):
# 	cv2.imshow(str(i), lp2[i])
# 	print( lp2[i].shape )

# LS = []
# for la, lb in zip(lp1, lp2):
# 	ls = np.zeros( la.shape )
# 	ls = cv2.add(ls, la[:, 0:col/2, :])
# Now add left and right halves of images in each level
# For half half
LS = []
for la,lb in zip(lp1, lp2):
	rows, cols, dpt = la.shape
	ls = np.hstack((la[:,0:int(cols/2),:], lb[:, int(cols/2):,:]))
	LS.append(ls)

# for i in range(0, 5):
	# cv2.imshow( str(i), LS[i] )

# To create the original image
expanded_image = cv2.pyrUp(LS[4])

for i in range(3, -1, -1):
	corrected_image = cv2.add(expanded_image, LS[i])
	# cv2.imshow("corrected", corrected_image)
	expanded_image = cv2.pyrUp(corrected_image)

# reconstructed image
cv2.imshow("corrected", corrected_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
