import os
import numpy as np
import cv2
from feature import feature_point_matching
from fundamental import get_fundamental_matrix
from util import draw_epilines

img1_path=os.path.join('dataset','Mesona1.JPG')
img2_path=os.path.join('dataset','Mesona2.JPG')
ratio=0.7
threshold=0.5
K=np.asarray([[1421.9, 0.5, 509.2],
              [0,   1421.9, 380.2],
              [0,        0,     1]])

if __name__=='__main__':

    img1=cv2.imread(img1_path)
    img2=cv2.imread(img2_path)
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # 1. get the correspondence across images
    keys1,keys2=feature_point_matching(img1,img2,ratio=ratio)

    # 2. get the Fundamental matrix by correspondence
    F,inlier_idxs=get_fundamental_matrix(keys1,keys2,threshold=threshold)
    F = F.T
    E = K.T @ F @ K
    inlier1 = keys1[inlier_idxs]
    inlier2 = keys2[inlier_idxs]

    # 3. draw epipolar lines
    print(f'# correspondence: {len(keys1)}')
    print(f'# inliers: {len(inlier_idxs)}')
    draw_epilines(img1, img2, inlier1, inlier2, F, 'epilines.png')