# import sys
# import cv2
# import numpy as np

# # https://www.codespeedy.com/convert-rgb-to-binary-image-in-python/

# # resize image
# def resize(image):
# 	dim = (900, 500)
# 	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
# 	return resized

# def print_image(image, name):
# 	cv2.imshow(name, image)
# 	print("shape of the image", image.shape)
# 	cv2.waitKey()
# 	return 0

# def Sobel(image, k_size):
# 	sobelx = cv2.Sobel(resized, cv2.CV_16S, 1, 0, ksize=7)
# 	sobely = cv2.Sobel(resized, cv2.CV_16S, 0, 1, ksize=7)
# 	sobel = cv2.add(sobelx, sobely)
# 	return sobel

# def laplacian(image):
# 	laplacian=cv2.Laplacian(image, cv2.CV_8U)
# 	return laplacian

# image = cv2.imread(sys.argv[1],0)
# print_image(image, "Original")
# resized = resize(image)
# print_image(resized, "resized")

# ret, bw_img = cv2.threshold(resized, 190 ,255,cv2.THRESH_BINARY)
# print_image(bw_img, "threshold")
# # sobel = Sobel(bw_img, 7)
# # laplacian = laplacian(bw_img)
# # print_image(sobel, "Sobel")
# kernel = np.array([[0,0,0,0,0,0,0],[0,-1,-1,-1,-1,-1,0],[0,-1,0,0,0,-1,0],[0,-1,0,0,0,-1,0],[0,-1,-1,-1,-1,-1,0],[0,0,0,0,0,0,0]])

# output_image = cv.morphologyEx(bw_img, cv.MORPH_HITMISS, kernel)
# print(output_image, "output")
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import sys
from skimage.exposure import rescale_intensity


def show_histogram(img):
    count,bins = np.histogram(img,range(257))
    plt.bar(bins[:-1]-0.5,count,width=1,edgecolor='none')
    plt.xlim([-0.5,255.5])
    plt.ylabel("pixel counts")
    plt.yticks(rotation=45)
    plt.xlabel('pixel range')
    plt.title("Image Histogram")
    plt.show()

def show_img(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    return 0

def resize(image):
	dim = (900, 500)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized


'''
function for finding rectangle shapes
'''
def segment_rect(img,gray_img,retrivel=None,method=None,min_area=None,max_area=None):
    gray = img.copy()
    gray2 = gray_img.copy()
    mask = np.zeros(gray.shape,np.uint8)

	#finding contours
    contours, hier = cv2.findContours(gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

		#it will returns ((x,y),(w,h),theta) where x,y is center of bounding box and w and h is width and height and theta is rortation
        rect = cv2.minAreaRect(cnt)
        w  =rect[1][0]
        h  = rect[1][1]
        a = w*h
        theta =rect[2]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # if(len(cnt)>=3):
        #     area=cv2.contourArea(cnt)

		#trying to threshholding the geometric properties of boxes such as width,height and area
        if a>=5000 and a<=70000 and (theta>-5 and theta<5) and (w>200 and w<750) and h>50:
            ellipse = cv2.fitEllipse(cnt)
            (center,axes,orientation) = ellipse
            majoraxis_length = max(axes)
            minoraxis_length = min(axes)
            eccentricity = np.sqrt(1-(minoraxis_length/majoraxis_length)**2)
            area=cv2.contourArea(cnt)
            equi_diameter = np.sqrt(4*area/np.pi)
            compactness=equi_diameter/majoraxis_length
            # if(eccentricity<=0.1 or eccentricity >=1) or (compactness <5):
            cv2.drawContours(gray2,[box],0,(0,255,0),3)
            cv2.drawContours(mask,[box],0,255,-1)
    show_img('rd',resize(gray2))
    return mask,gray2


'''
	function for binarizing image where white regions are object(or edges of object) and 
	black color is background, try to make the edges of those fields continous
	so that there should not be any hole, that should be perfect rectangle which will
	beinfit the further process to choose correct rectangle.
'''

def preprocess_img(img):
    gray = img.copy()

    #to detect horizontal lines
    stel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1,7))

    #to detect vertical lines
    stel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,1))

    #to remove blobs
    stel3 = np.ones((11,11))
    stel4 = np.ones((9,9))
    
    #thresholding
    thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,71,1)

    #erode
    # res=cv2.morphologyEx(thresh2,cv2.MORPH_ERODE,stel1)
    # res2=cv2.morphologyEx(res,cv2.MORPH_ERODE,stel2)

    
    # thresh2 = cv2.morphologyEx(thresh2,cv2.MORPH_CLOSE,stel3)
    thresh2 = cv2.bitwise_not(thresh2)
    thresh2 = cv2.erode(thresh2,stel4)
    thresh2 = cv2.dilate(thresh2,stel4)
    # thresh2 = cv2.erode(thresh2,stel4)
    t = resize(thresh2)
    show_img('thresh2' ,t)
    return thresh2

def imshow_components(labels):

    hue = np.uint8(179*labels/np.max(labels))
    blank_channel = 255*np.ones_like(hue)
    labeled_img = cv2.merge([hue, blank_channel, blank_channel])

    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    labeled_img[hue==0] = 0
    return labeled_img

def character_detection(imposed_img,field_img):
    '''
    @imposed_img - image extracted charcater after imposing fields on true image
    @field)_img - form_field segmentation image

    returns labelled image in rgb colored 
    '''
    field_img  = cv2.adaptiveThreshold(field_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    show_img('f',resize(field_img))
    imposed_img = cv2.GaussianBlur(imposed_img,(5,5),sigmaX=0)
    imposed_img = cv2.adaptiveThreshold(imposed_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    resultant = imposed_img-field_img
    resultant = cv2.erode(resultant,np.ones((3,3)))
    resultant = cv2.dilate(resultant,np.ones((3,3)))
    show_img('res',resize(resultant))
    ret,labels = cv2.connectedComponents(resultant)
    print(ret)
    print(labels)
    labeled_img = imshow_components(labels)
    return labeled_img

if __name__ == "__main__":

    #input image
    img = sys.argv[1]
    image = cv2.imread(img)

    #convrted into rgb image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),sigmaX=0)
    gray2 = gray.copy()

    show_img('gray',resize(gray))

    pre_img = preprocess_img(gray)
    field_img,g = segment_rect(pre_img,image)

    #to get labelled image of fields
    ret,labels =cv2.connectedComponents(field_img)
    imshow_components(labels)
    fimg = resize(field_img)
    show_img('fimg',fimg)
    mask = rescale_intensity(field_img,in_range=(0,255),out_range=(0,1))

    #character extraction using field image
    imposed_img = gray*mask
    show_img('new',resize(imposed_img))

    #character_detection
    cdet = character_detection(imposed_img,field_img)
    show_img('characters',resize(cdet))
