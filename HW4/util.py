import numpy as np
import cv2
import matplotlib.pyplot as plt


def to_homogeneous(pts):
    one = np.ones((pts.shape[0],1))
    return np.hstack((pts,one))

def norm_line(lines):
    a = lines[:,0:1]
    b = lines[:,1:2]
    length = np.sqrt(a**2 + b**2)
    return lines / length

def drawlines(img1, img2, lines, pts1, pts2):
    '''
    :param img1: image on which we draw the epilines for the points in img2
    :param lines: corresponding epilines
    '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def draw_epilines(gray1, gray2, inlier1, inlier2, F, filename):
    # (F.T @ to_homogeneous(inlier2).T).T
    lines1_unnorm = to_homogeneous(inlier2) @ F
    lines1 = norm_line(lines1_unnorm)
    img5, img6 = drawlines(gray1, gray2, lines1, inlier1.astype(np.int32), inlier2.astype(np.int32))

    lines2_unnorm = (F @ to_homogeneous(inlier1).T).T
    lines2 = norm_line(lines2_unnorm)
    img3, img4 = drawlines(gray2, gray1, lines2, inlier2.astype(np.int32), inlier1.astype(np.int32))

    plt.subplot(221), plt.imshow(img5)
    plt.subplot(222), plt.imshow(img6)
    plt.subplot(223), plt.imshow(img4)
    plt.subplot(224), plt.imshow(img3)
    plt.savefig(filename)