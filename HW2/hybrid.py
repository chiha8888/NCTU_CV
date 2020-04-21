import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
manager = plt.get_current_fig_manager()
manager.window.showMaximized()


def hybrid(img1,img2,cutoff_frequency,Filter):
    assert img1.shape==img2.shape,'shape not match'
    h,w,c=img1.shape
    lowPassed = convolution(img1, Filter(h,w,cutoff_frequency,lowPass=True))
    highPassed = convolution(img2, Filter(h,w,cutoff_frequency, lowPass=False))
    return highPassed+lowPassed,lowPassed,highPassed

def idealFilter(h,w,cutoff_frequency,lowPass):
    """
    :return: (h,w) ndarray
    """
    x0,y0=w//2,h//2
    if lowPass:
        H=np.zeros((h,w))
        for x in range(x0-cutoff_frequency, x0+cutoff_frequency):
            # (x-x0)^2 + (y-y0)^2 = r^2
            for y in range(int(y0-math.sqrt(cutoff_frequency**2-(x-x0)**2)),int(y0+math.sqrt(cutoff_frequency**2-(x-x0)**2))):
                    H[y,x]=1
    else:
        H=np.ones((h,w))
        for x in range(x0-cutoff_frequency,x0+cutoff_frequency):
            for y in range(int(y0-math.sqrt(cutoff_frequency**2-(x-x0)**2)),int(y0+math.sqrt(cutoff_frequency**2-(x-x0)**2))):
                    H[y,x]=0
    return H

def GaussianFilter(h,w,cutoff_frequency, lowPass):
    """
    :return: (h,w) ndarray
    """
    x0,y0=w//2,h//2
    if lowPass:
        H=np.zeros((h,w))
        for x in range(w):
            for y in range(h):
                H[y,x]=math.exp(-1*((x-x0)**2+(y-y0)**2)/(2*cutoff_frequency**2))
    else:
        H=np.ones((h,w))
        for x in range(w):
            for y in range(h):
                H[y,x]-=math.exp(-1*((x-x0)**2+(y-y0)**2)/(2*cutoff_frequency**2))
    return H

def convolution(image, H):
    """
    :param image: (H,W,C) ndarray
    :param H: (H,W) filter ndarray
    """
    h,w,c=image.shape
    image_sp = np.zeros((h,w,c))
    result = np.zeros((h,w,c))
    for channel in range(c):
        image_ = image[:,:,channel] / 255
        """
        Multiply the input image by (-1)^x+y to center the transform.
        """
        for i in range(h):
            for j in range(w):
                image_[i,j]=((-1)**(i+j))*image_[i,j]
        """
        Compute ftt F for input image
        """
        F=(np.fft.fft2(image_))
        image_sp[:,:,channel]=20*np.log(np.abs(F))
        """
        Multiply F by H, compute the inverse ftt of the result
        """
        result[:,:,channel]=np.absolute(np.fft.ifft2(F*H))
    return result

def normalize(img):
    """
    normalize img to 0~1
    """
    img_min = img.min()
    img_max = img.max()
    return (img - img_min) / (img_max - img_min)

if __name__ == "__main__":
    root = os.path.join('hw2_data','task1and2_hybrid_pyramid')
    name1='6_makeup_after.jpg'
    name2='6_makeup_before.jpg'
    cutoff_frequencies=[6] # bigger value -> clearer lowPass & less highPass
    # load images
    img1 = cv2.imread(os.path.join(root,name1))
    img2 = cv2.imread(os.path.join(root,name2))
    if name1=='6_makeup_after.jpg' and name2=='6_makeup_before.jpg':
        img1=img1[:-1,:-1,:]

    for cutoff_frequency in cutoff_frequencies:
        for name,Filter in zip(['ideal','gaussian'],[idealFilter,GaussianFilter]):
            result,lowPassed,highPassed = hybrid(img1,img2,cutoff_frequency,Filter)

            plt.subplot(231), plt.imshow(img1[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title('image1')
            plt.subplot(234), plt.imshow(img2[:, :, ::-1]), plt.xticks([]), plt.yticks([]), plt.title('image2')
            plt.subplot(232), plt.imshow(normalize(lowPassed)[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title('lowPass')
            plt.subplot(235), plt.imshow(normalize(highPassed)[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title('highPass')
            plt.subplot(133), plt.imshow(normalize(result)[:,:,::-1]), plt.xticks([]), plt.yticks([]), plt.title(f'result (cutoff frequency={cutoff_frequency})')
            plt.show()

            cv2.imwrite(os.path.join('result', 'task1', name,
                                     f'{name1.split(".")[0]}_{name2.split(".")[0]}_cutoff{cutoff_frequency}.jpg'),
                        normalize(result) * 255)