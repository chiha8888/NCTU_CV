import os
import cv2
import numpy as np
import collections
from scipy.spatial.distance import cdist

def loadImgs(filepath,size):
    img_list=[]

    class_names=os.listdir(filepath)
    for class_name in class_names:
        class_path = os.path.join(filepath,class_name)
        image_names = os.listdir(class_path)
        for image_name in image_names:
            try:
                img_path = os.path.join(class_path,image_name)
                img = cv2.imread(img_path)
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                # img=cv2.resize(img,size)
                img_list.append(img)
            except:
                pass

    return img_list

def get_all_features(img_list):
    """
    :param imgs: a list of images
    :return: (#keypoints from all images,128) ndarray
    """
    sift = cv2.xfeatures2d.SIFT_create()
    describes=None
    for img in img_list:
        kp,des=sift.detectAndCompute(img,None)
        if des is not None:
            describes=np.vstack((describes,des)) if describes is not None else des

    return describes

def img2histogram(img,centers):
    """represent img by frequencies of visual words
    :param img: an (200x200) ndarray img
    :param centers: (k=200,128) center points
    :return: (200)
    """
    cluster=len(centers)
    sift = cv2.xfeatures2d.SIFT_create()
    kp,des=sift.detectAndCompute(img,None)
    distances=cdist(des,centers,'euclidean')
    counter=collections.Counter(np.argmin(distances,axis=1))
    re=np.zeros(cluster)
    for i in counter:
        re[i]=counter[i]

    return re

def get_histograms(img_list,cluster,centers):
    histograms=np.zeros((len(img_list),cluster))
    for i,img in enumerate(img_list):
        histograms[i]=img2histogram(img,centers)

    return histograms

def knn(indices):
    """
    :param indics: (#test images,k) ndarray
    """
    indices = indices // 100
    k=indices.shape[1]
    acc=0
    for i in range(len(indices)):
        target_class=i//10
        predict_class=collections.Counter(indices[i]).most_common(1)[0][0]
        if target_class==predict_class:
            acc+=1

    return acc/len(indices)

train_path=os.path.join('hw5_data','train')
test_path=os.path.join('hw5_data','test')
size=(200,200)
cluster=80
k=5

if __name__=='__main__':
    # load images
    train_imgs=loadImgs(train_path,size=size)
    test_imgs=loadImgs(test_path,size=size)
    # k-means
    all_features=get_all_features(train_imgs)


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _,_,centers = cv2.kmeans(all_features,cluster,None,criteria,10,cv2.KMEANS_PP_CENTERS)

    # train histograms
    train_histograms=get_histograms(train_imgs,cluster,centers)
    # test histograms
    test_histograms=get_histograms(test_imgs,cluster,centers)

    indices=np.argsort(cdist(test_histograms,train_histograms,'euclidean'),axis=1)

    acc=knn(indices[:,:k])
    print(acc)

