import os
import numpy as np
import cv2
from feature import feature_point_matching
from fundamental import get_fundamental_matrix
from util import draw_epilines,plot
from triangulation import compute_P_from_essential,threeD_from_camera_matrix,choose_best_threeD
import matlab.engine
import matlab


img1_path=os.path.join('dataset','Mesona1.JPG')
img2_path=os.path.join('dataset','Mesona2.JPG')
ratio=0.5
threshold=0.05
K1=K2=np.asarray([[1421.9, 0.5, 509.2],
              [0,   1421.9, 380.2],
              [0,        0,     1]])

def ndarray2matlab(x):
    return matlab.double(x.tolist())

if __name__=='__main__':

    img1=cv2.imread(img1_path)
    img2=cv2.imread(img2_path)
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # 1. get the correspondence across images
    keys1,keys2 = feature_point_matching(img1,img2,ratio=ratio)

    # 2. get the Fundamental matrix by correspondence
    F,inlier_idxs = get_fundamental_matrix(keys1,keys2,threshold=threshold)
    inlier1 = keys1[inlier_idxs]
    inlier2 = keys2[inlier_idxs]

    # 3. draw epipolar lines
    print(f'# correspondence: {len(keys1)}')
    print(f'# inliers: {len(inlier_idxs)}')
    draw_epilines(img1, img2, inlier1, inlier2, F, 'epilines.png')

    # 4. four possible P2
    E = K1.T @ F @ K2
    P1 = np.hstack((np.eye(3),np.zeros((3,1)))) # first camera matrix
    P2s= compute_P_from_essential(E)  # second camera matrix

    # 5. four possible 3D points from P1 & P2
    threeDs=threeD_from_camera_matrix(K1,K2,P1,P2s,inlier1,inlier2)

    # 6. find the most appropriate
    threeD=choose_best_threeD(threeDs,P2s)
    plot(threeD)

    # 7. call matlab
    eng = matlab.engine.start_matlab()
    eng.obj_main(ndarray2matlab(threeD), ndarray2matlab(threeD), ndarray2matlab(K1@P1), img1_path, matlab.int32([1]), nargout=0)
    eng.quit()