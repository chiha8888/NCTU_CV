import numpy as np
import cv2


def hybrid(img1,img2,Filter):
    lowPassed = filterDFT(img1, Filter(5, highPass=False))
    highPassed = img2/255 - filterDFT(img2, Filter(5, highPass=False)) # img - loss pass = highpass
    return highPassed + lowPassed

def GaussianFilter(frequency, highPass=True):
    # filter size
    size = 8 * frequency + 1
    if not size % 2:
        size = size + 1
    # lambda function for gaussian at i,j position
    gaussian = lambda i,j: np.exp(-1.0 * ((i - size//2)**2 + (j - size//2)**2) / (2 * frequency**2))
    k = np.array([[1-gaussian(i,j) if highPass else gaussian(i,j) for j in range(size)] for i in range(size)])
    return k/np.sum(k)

def idealFilter(frequency, highPass=True):
    # filter size
    size = 8 * frequency + 1
    if not size % 2:
        size = size + 1
    # lambda function for gaussian at i,j position
    ideal = lambda i,j:((i - size//2)**2+(j - size//2)**2)**0.5
    if highPass:
        return np.array([[0 if ideal(i,j) <= frequency else 1 for j in range(size)] for i in range(size)])
    else:
        return np.array([[1 if ideal(i,j) <= frequency else 0 for j in range(size)] for i in range(size)])

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
    root = "hw2_data/task1and2_hybrid_pyramid"
    # load images
    img1 = cv2.imread(root+"/3_cat.bmp")
    img2 = cv2.imread(root+"/3_dog.bmp")
    resultGaussian = hybrid(img1,img2,GaussianFilter)*255
    resultideal = hybrid(img1,img2,idealFilter)
    cv2.imwrite("Gaussian_result0.jpg",resultGaussian)
    cv2.imwrite("ideal_result0.jpg",resultideal)
    