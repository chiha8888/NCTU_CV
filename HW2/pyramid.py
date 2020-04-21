import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
# to do draw spectrum

def GaussianFilter(frequency, highPass=False):
    # filter size
    size = 8 * frequency + 1
    if not size % 2:
        size = size + 1
    # lambda function for gaussian at i,j position
    gaussian = lambda i,j: np.exp(-1.0 * ((i - size//2)**2 + (j - size//2)**2) / (2 * frequency**2))
    k = np.array([[1-gaussian(i,j) if highPass else gaussian(i,j) for j in range(size)] for i in range(size)])
    return k/np.sum(k)

def filterDFT(img, filterH):
    k_h, k_w = filterH.shape[0],filterH.shape[1]
    start_h,start_w = (img.shape[0] - k_h) // 2, (img.shape[1] - k_w) // 2
    pad_filter = np.zeros(img.shape[:2])
    pad_filter[start_h : start_h + k_h, start_w : start_w + k_w] = filterH # pad the filter

    # ft the filter
    filt_fft = np.fft.fft2(pad_filter)
    # RGB
    if len(img.shape) == 3:
        result = np.zeros(img.shape)
        # for RGB
        for color in range(3):
            img_fft = np.fft.fft2(img[:, :, color])
            result[:, :, color] = np.fft.fftshift(np.fft.ifft2(img_fft * filt_fft)).real # apply the filter
        return result
    else: # gray
        img_fft = np.fft.fft2(img)
        result_img = np.fft.ifft2(img_fft * filt_fft).real # apply the filter
        return np.fft.fftshift(result_img)

def subsampling(img):
    newImg = np.zeros((img.shape[0]//2,img.shape[1]//2,3)) if len(img.shape) == 3 else np.zeros((img.shape[0]//2,img.shape[1]//2))
    # for each row and column pick the even pixel
    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            newImg[i][j] = img[2*i][2*j]
    return newImg
def upsampling(img,old_result):
    # calculate the difference of size between upsampling image and origin image
    padc, padr = np.array(old_result.shape[:2]) - np.array(img.shape[:2])*2
    # column interpolation
    col_idx = (np.ceil(np.arange(1, 1 + img.shape[0]*2)//2) - 1).astype(int)
    # row interpolation 
    row_idx = (np.ceil(np.arange(1, 1 + img.shape[1]*2)//2) - 1).astype(int)
    result = img[:, row_idx][col_idx, :]
    # if RGB
    if len(img.shape) == 3:
        # pad 0 to match the size of old image
        return np.pad(result,((padc,0),(padr,0),(0,0)),"constant")
    else: # gray
        return np.pad(result,((padc,0),(padr,0)),"constant")

## convert img to spectrum
def img_to_spectrum(img):
    # if RGB
    if len(img.shape) == 3:
        result = np.zeros(img.shape)
        for color in range(3):
            result[:, :, color] = np.fft.fft2(img[:, :, color]).real
        return normalize(np.log(1+np.abs(np.fft.fftshift(result)))) # normalizing result
    else: # gray
        return normalize(np.log(1+np.abs(np.fft.fftshift(np.fft.fft2(img))))) # normalizing result

def normalize(img):
    """
    normalize img to 0~1
    """
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)

if __name__ == "__main__":
    root = os.path.join('hw2_data','task1and2_hybrid_pyramid')
    name='4_einstein.bmp'
    step=4
    img1 = cv2.imread(os.path.join(root,name))
    result = img1
    for i in range(step):
        old_result = result
        result = filterDFT(result,GaussianFilter(2))
        cv2.imwrite(os.path.join('result','task2',f'gaussian_{i}.jpg'),result)

        gaussian_spectrum = img_to_spectrum(result)
        plt.imshow(gaussian_spectrum)
        plt.savefig(os.path.join('result','task2',f'gaussian_sp_{i}.jpg'))

        if i == 4:
            Laplacian = result
        else:
            result = subsampling(result)
            result2 = upsampling(result,old_result)
            Laplacian = old_result - result2
        cv2.imwrite(os.path.join('result','task2',f'Laplacian_{i}.jpg'),Laplacian)

        Laplacian_spectrum = img_to_spectrum(Laplacian)
        plt.imshow(Laplacian_spectrum)
        plt.savefig(os.path.join('result','task2',f'Laplacian_sp_{i}.jpg'))
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

        ## openCV result
        # A = cv2.imread("img.jpg")
        # B = cv2.imread("img.jpg")
        # # generate Gaussian pyramid for A
        # G = A.copy()
        # gpA = [G]
        # for i in range(4):
        #     G = cv2.pyrDown(G)
        #     gpA.append(G)

        # # generate Gaussian pyramid for B
        # G = B.copy()
        # gpB = [G]
        # for i in range(4):
        #     G = cv2.pyrDown(G)
        #     gpB.append(G)

        # # generate Laplacian Pyramid for A
        # lpA = [gpA[3]]
        # for i in range(3,0,-1):
        #     GE = cv2.pyrUp(gpA[i])
        #     L = cv2.subtract(gpA[i-1],GE)
        #     plt.imshow(img_to_spectrum(L))
        #     plt.show()
        #     lpA.append(L)

        # # generate Laplacian Pyramid for B
        # lpB = [gpB[3]]
        # for i in range(3,0,-1):
        #     GE = cv2.pyrUp(gpB[i])
        #     L = cv2.subtract(gpB[i-1],GE)
        #     lpB.append(L)

        # # Now add left and right halves of images in each level
        # LS = []
        # for la,lb in zip(lpA,lpB):
        #     rows,cols,dpt = la.shape
        #     ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
        #     LS.append(ls)

        # # now reconstruct
        # ls_ = LS[0]
        # for i in range(1,4):
        #     ls_ = cv2.pyrUp(ls_)
        #     ls_ = cv2.add(ls_, LS[i])

        # # image with direct connecting each half
        # real = np.hstack((A[:,:cols//2],B[:,cols//2:]))

        # cv2.imwrite('Pyramid_blending2.jpg',ls_)
        # cv2.imwrite('Direct_blending.jpg',real)