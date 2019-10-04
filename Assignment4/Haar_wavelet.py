import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from pywt import dwt2, idwt2
import sys

gray_conversion = lambda rgb : np.dot(rgb[...,:3],[0.299 , 0.587, 0.114])

yuv_from_rgb = np.array([[ 0.299     ,  0.587     ,  0.114      ],
                         [-0.14714119, -0.28886916,  0.43601035 ],
                         [ 0.61497538, -0.51496512, -0.10001026 ]])

rgb_from_yuv = np.linalg.inv(yuv_from_rgb)

def psnr(I1,I2):
    if(len(I1.shape)==2 and len(I2.shape)==2):
        MSE = np.mean(np.power((I1-I2),2),dtype=np.float64)
        if(np.max(I1) > 1):
            R=255
        else:
            R=1
        psnr_val =10*np.log10(R**2/MSE)
    return psnr_val


def run_length_encoding(image):
    bitstream = ""
    skipped_zeros = 0
    for i in range(image.shape[0]):
        if(image[i]!=0):
            bitstream = bitstream + str(image[i]) + " "+ str(skipped_zeros)+ " "
            skipped_zeros=0
        else:
            skipped_zeros=skipped_zeros+1
    return bitstream


def gaussian_noise(img,var=0.001,mean=0):
    image = img.copy()
    if len(image.shape)==2:
        m,n=image.shape
        sigma = var**0.5
        gaussian_matrix = np.random.normal(mean,sigma,(m,n))
        noise_image = image+gaussian_matrix
    if len(image.shape)>=3:
        noise_image = np.copy(image)
        sigma = var**0.5
        gaussian_matrix = np.random.normal(mean,sigma,(image.shape[0],image.shape[1]))
        for i in range(image.shape[2]):
            noise_image[:,:,i]=image[:,:,i]+gaussian_matrix
    return noise_image,gaussian_matrix

def haar2D(image):
    img = image.copy()
    assert len(img.shape)==2
    rows = img.shape[0]
    col= img.shape[1]
    res = img.copy()

    for i in range(rows):
        l = 0
        m = col//2
        for j in range(col//2):
            res[i][l] = (img[i,2*j] + img[i,2*j+1])/np.sqrt(2)
            l= l+1
            res[i][m] = (img[i,2*j]-img[i,2*j+1])/np.sqrt(2)
            m=m+1

    result_img = np.zeros(res.shape)
    for i in range(col):
        for j in range(rows//2):
            result_img[j][i] = (res[2*j][i] +res[2*j+1][i])/np.sqrt(2) 
            result_img[rows//2+j][i] = (res[2*j][i] - res[2*j+1][i])/np.sqrt(2)

    #approximation
    LL = result_img[:rows//2,:col//2]

    #details coef- Horizontal
    LH= result_img[:rows//2,col//2:]

    #details coef - Vertical
    HL = result_img[rows//2:,:col//2]

    #details coef - Diagonal
    HH= result_img[rows//2:,col//2:]
            
    coefficients = dict()
    coefficients["LH"] = LH
    coefficients["HL"] = HL
    coefficients["HH"] = HH
    return result_img,LL,coefficients


def haar_transform(image,levels=None,threshold=None): 
    img = image.copy()
    m,n = img.shape
    detail_coef = dict()
    if(levels is not None):
        assert np.power(2,levels)<=m
    haar_result,LL,_coef  = haar2D(img)
    m=m//2
    n = n//2
    level =1
    detail_coef["level_"+str(level)] = _coef
    if(m==n):
        if(levels is not None):     
            while(level!=levels):
                res,L,_coef = haar2D(LL)
                haar_result[:m,:n] = res
                LL = L
                m = m//2
                n = n//2
                level = level+1
                if(threshold is not None):
                    for key,value in _coef.items():
                        _coef[key] = np.where(np.abs(_coef[key])<threshold,0,_coef[key])
                detail_coef["level_"+str(level)] = _coef
        else:
            while(m!=1):
                res,L,_coef = haar2D(LL)
                haar_result[:m,:n] = res
                LL = L
                m = m//2
                n = n//2
                level = level+1
                if(threshold is not None):
                    for key,value in _coef.items():
                        _coef[key] = np.where(np.abs(_coef[key])<threshold,0,_coef[key])
                detail_coef["level_"+str(level)] = _coef

    haar_result = haar_result.astype(np.float32)
    if(threshold is not None):
        haar_result = np.where(np.abs(haar_result)<threshold,0,haar_result)
    return haar_result,LL,detail_coef


def inverseHaar2D(a,detail_coef,level=None,threshold=None):
    total_levels = len(detail_coef)
    k = total_levels
    for i in range(total_levels):
        detail_coef_level = detail_coef["level_"+str(k)]
        LL = a
  
        LH = detail_coef_level['LH']
        HL = detail_coef_level['HL']
        HH = detail_coef_level['HH']
        if(threshold is not None):
            LH = np.where(np.abs(LH)>threshold,LH,0)
            HL = np.where(np.abs(HL)>threshold,HL,0)
            HH = np.where(np.abs(HH)>threshold,HH,0)
        L = np.hstack((LL,LH))
        H = np.hstack((HL,HH))
        haar_t = np.vstack((L,H))

        res = np.zeros(haar_t.shape)
        for i in range(res.shape[1]):
            for j in range(res.shape[0]//2):
                res[2*j][i] = (haar_t[j][i]+haar_t[res.shape[0]//2+j][i])/np.sqrt(2)
                res[2*j+1][i] = (haar_t[j][i]-haar_t[res.shape[0]//2+j][i])/np.sqrt(2)

        new_approx = np.zeros((res.shape[0],res.shape[1])) 
        for i in range(res.shape[0]):
            for j in range(res.shape[1]//2):
                new_approx[i][2*j] = (res[i][j]+res[i][(res.shape[1]//2)+j])/np.sqrt(2)
                new_approx[i][2*j+1] = (res[i][j]-res[i][(res.shape[1]//2)+j])/np.sqrt(2)
        a = new_approx
        k = k-1

    return new_approx

def yuv2rgb(yuv):
    return np.clip(np.dot(yuv,rgb_from_yuv),0,1)


if __name__ == "__main__":
    img_file = sys.argv[1]
    orig_img  = plt.imread(img_file)
    if(len(orig_img.shape)>=3):
        orig_img = orig_img[:,:,:3]
        YUV = np.dot(orig_img,yuv_from_rgb)
        orig_img = YUV[:,:,0]

    # orig_noised_img,_ = gaussian_noise(orig_img)

    #forward wavelet transform
    haar_result,a,detail_coef = haar_transform(orig_img)
    cv2.imshow("haar",haar_result)
    cv2.waitKey(0)

    # #run length encoding
    # bitstream = run_length_encoding(haar_result.flatten())

    # #writing encoded stream to file
    # with open("encoded_file.txt","w") as f:
    #     f.write(bitstream)

    # #inverse wavelet transform (threshold to remove noise)
    # inv_img = inverseHaar2D(a,detail_coef)
    # YUV[:,:,0] = inv_img
    # back_rgb = yuv2rgb(YUV)
    # plt.imsave(img_file[:-4]+"_inverse_haar.png",back_rgb)
    # plt.imshow(back_rgb)
    # plt.show()
