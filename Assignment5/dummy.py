import numpy as np 
import cv2
import matplotlib.pyplot as plt
import sys
from skimage.exposure import rescale_intensity

def getFFTspectrum(img,a):
    K = np.fft.fft2(img,norm='ortho')
    Kshift = np.fft.fftshift(K)
    magnitude_spec =a*np.log(np.abs(Kshift)) 
    return magnitude_spec,K

def intensity_rescale(image):
    return rescale_intensity(image,in_range=(image.min(),image.max()),out_range=(0,1))
if __name__ == "__main__":
    file = sys.argv[1]
    img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)

    fft,_ = getFFTspectrum(img,20)
    fft = cv2.resize(intensity_rescale(fft),(400,400))
    # plt.imshow(fft,cmap='gray')
    # plt.show()
    cv2.imshow('fft',fft)
    cv2.waitKey(0)
