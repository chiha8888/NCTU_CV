import matplotlib.pyplot as plt
import numpy as np
import cv2

# normalized cross correlation
def ncc(g,f):
    g=g-g.mean(axis=0)
    f=f-f.mean(axis=0)
    return np.sum((g * f)/(np.linalg.norm(g)) * (np.linalg.norm(f)))
def Align(target,x,t):
    mini = float("-inf")
    col=np.linspace(-t,t,2*t,dtype=int)
    row=np.linspace(-t,t,2*t,dtype=int)
    for i in col:
        for j in row:
            # find ncc difference
            diff = ncc(target,np.roll(x,[i,j],axis=(0,1)))
            # if difference bigger than found before
            if diff > mini:
                mini = diff
                offset = [i,j]
    return offset
if __name__ == "__main__":

    root = "hw2_data/task3_colorizing/"
    name = 'tobolsk.jpg'
    img=cv2.imread(root+name,0)
    w,h=img.shape[:2]
    img=img[int(w*0.01):int(w-w*0.01),int(h*0.01):int(h-h*0.01)] # remove the 
    w,h=img.shape[:2]

    # get RGB image
    height=w//3
    blue=img[0:height,:]
    green=img[height:2*height,:]
    red=img[2*height:3*height,:]

    # get the offset of x and y direction
    offect_g = Align(blue,green,10)
    offset_r = Align(blue,red,10)

    # shift the green and red image
    green=np.roll(green,offect_g,axis=(0,1))
    red=np.roll(red,offset_r,axis=(0,1))

    # concat RGB and draw
    result = np.concatenate((red[:,:,None],green[:,:,None],blue[:,:,None]),axis=2)
    plt.imshow(result)
    plt.savefig("result_{}.jpg".format(name))