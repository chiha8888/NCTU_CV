import os
import numpy as np
import cv2


def hybrid(img1,img2,Filter):
    lowPassed = filterDFT(img1, Filter(5, highPass=False))
    highPassed = img2/255 - filterDFT(img2, Filter(5, highPass=False)) # img - loss pass = highpass
    return highPassed + lowPassed

def GaussianFilter(cutoff_frequency, highPass=True):
    # filter size
    size = 8 * cutoff_frequency + 1
    if not size % 2:
        size = size + 1
    # lambda function for gaussian at i,j position
    gaussian = lambda i,j: np.exp(-1.0 * ((i - size//2)**2 + (j - size//2)**2) / (2 * cutoff_frequency**2))
    k = np.array([[1-gaussian(i,j) if highPass else gaussian(i,j) for j in range(size)] for i in range(size)])
    return k/np.sum(k)

def idealFilter(cutoff_frequency, highPass=True):
    # filter size
    size = 8 * cutoff_frequency + 1
    if not size % 2:
        size = size + 1
    # lambda function for gaussian at i,j position
    ideal = lambda i,j:((i - size//2)**2+(j - size//2)**2)**0.5
    if highPass:
        return np.array([[0 if ideal(i,j) <= cutoff_frequency else 1 for j in range(size)] for i in range(size)])
    else:
        return np.array([[1 if ideal(i,j) <= cutoff_frequency else 0 for j in range(size)] for i in range(size)])

def filterDFT(img, filterH):
    k_h, k_w = filterH.shape[0],filterH.shape[1]
    start_h,start_w = (img.shape[0] - k_h) // 2, (img.shape[1] - k_w) // 2
    pad_filter = np.zeros(img.shape[:2])
    pad_filter[start_h : start_h + k_h, start_w : start_w + k_w] = filterH # pad the filter

    filt_fft = np.fft.fft2(pad_filter)
    # RGB
    if len(img.shape) == 3:
        result = np.zeros(img.shape)
        for color in range(3):
            img_fft = np.fft.fft2(img[:, :, color])
            result[:, :, color] = np.fft.fftshift(np.fft.ifft2(img_fft * filt_fft)).real # apply the filter
        return result/255
    else: # gray
        img_fft = np.fft.fft2(img)
        result_img = np.fft.ifft2(img_fft * filt_fft).real
        return np.fft.fftshift(result_img)/255


if __name__ == "__main__":
    root = os.path.join('hw2_data','task1and2_hybrid_pyramid')
    name1='1_bicycle.bmp'
    name2='1_motorcycle.bmp'
    # load images
    img1 = cv2.imread(os.path.join(root,name1))
    img2 = cv2.imread(os.path.join(root,name2))

    resultGaussian = hybrid(img1,img2,GaussianFilter)*255
    resultideal = hybrid(img1,img2,idealFilter)

    cv2.imshow('gaussian',resultGaussian/255)
    cv2.imshow('ideal',resultideal/255)
    cv2.imwrite(os.path.join('result','task1',f'gaussian_{name1.split(".")[0]}_{name2.split(".")[0]}.jpg'),resultGaussian)
    cv2.imwrite(os.path.join('result','task1',f'ideal_{name1.split(".")[0]}_{name2.split(".")[0]}.jpg'),resultideal)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()