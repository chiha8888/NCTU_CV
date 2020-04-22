import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# normalized cross correlation
def ncc(g,f):
    g=g-g.mean(axis=0)
    f=f-f.mean(axis=0)
    return np.sum((g * f)/(np.linalg.norm(g)) * (np.linalg.norm(f)))

def Align(img1,img2,t):
    mini = float("-inf")
    col=np.linspace(-t,t,2*t,dtype=int)
    row=np.linspace(-t,t,2*t,dtype=int)
    for i in col:
        for j in row:
            # find ncc difference
            diff = ncc(img1,np.roll(img2,[i,j],axis=(0,1)))
            # if difference bigger than found before
            if diff > mini:
                mini = diff
                offset = [i,j]
    return offset

def subsampling(img,rate):
    newImg = np.zeros((img.shape[0]//rate,img.shape[1]//rate,3)) if len(img.shape) == 3 else np.zeros((img.shape[0]//rate,img.shape[1]//rate))
    # for each row and column pick the even pixel
    for i in range(newImg.shape[0]):
        for j in range(newImg.shape[1]):
            newImg[i][j] = img[2*i][2*j]
    return newImg

if __name__ == "__main__":

    root = os.path.join('hw2_data','task3_colorizing')
    name = 'onion_church.tif'
    IS_TIF=True if name.split('.')[-1]=='tif' else False
    subsample_rate=10
    t=15 # t depends on image
    img=cv2.imread(os.path.join(root,name),0)
    h, w = img.shape[:2]
    img = img[int(h * 0.02):int(h - h * 0.02), int(w * 0.02):int(w - w * 0.02)]  # the cutting ratio depends on image
    h,w = img.shape[:2]
    # get RGB image
    height = h // 3
    blue = img[0:height, :]
    green = img[height:2 * height, :]
    red = img[2 * height:3 * height, :]

    # Resize blue,green,red to smaller size before Align
    if IS_TIF:
        #subsample_img=subsampling(img,subsample_rate)
        subsample_img= cv2.resize(img, (w//subsample_rate, h//subsample_rate), interpolation=cv2.INTER_CUBIC)  # better subsample result
        height = subsample_img.shape[0] // 3
        blue_ = subsample_img[0:height, :]
        green_ = subsample_img[height:2 * height, :]
        red_ = subsample_img[2 * height:3 * height, :]

    # get the offset of x and y direction
    offset_g = Align(blue_ if IS_TIF else blue,green_ if IS_TIF else green,t)
    offset_r = Align(blue_ if IS_TIF else blue,red_ if IS_TIF else red,t)
    print(offset_g)
    print(offset_r)

    # shift the green and red image
    green=np.roll(green,[element*subsample_rate for element in offset_g] if IS_TIF else offset_g,axis=(0,1))
    red=np.roll(red,[element*subsample_rate for element in offset_r] if IS_TIF else offset_r,axis=(0,1))

    # concat RGB and draw
    result = np.concatenate((red[:,:,None],green[:,:,None],blue[:,:,None]),axis=2)
    plt.imshow(result)
    plt.savefig(os.path.join('result','task3',f'{name.split(".")[0]}'))
    plt.waitforbuttonpress(0)