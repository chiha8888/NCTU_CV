import numpy as np
import cv2


def img2keypointsandfeature(img):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, features = sift.detectAndCompute(img, None)
    keypoints = np.float32([i.pt for i in keypoints])
    return keypoints, features

def valid_matching(keypoints1, features1, keypoints2, features2, ratio):
    ##### opencv feature matching #####
    # match_instance = cv2.DescriptorMatcher_create("BruteForce")
    # All_Matches = match_instance.knnMatch(features1, features2, 2)
    # valid_matches = []
    # for val in All_Matches:
    #     if len(val) == 2 and val[0].distance < val[1].distance * ratio:
    #         valid_matches.append((val[0].trainIdx, val[0].queryIdx))
    # print(valid_matches)
    raw_match = []
    match_dist = []
    # find the closest and the second closest features
    for i in range(features1.shape[0]):
        if np.linalg.norm(features1[i] - features2[0]) < np.linalg.norm(features1[i] - features2[1]):
            closest = np.linalg.norm(features1[i] - features2[0])
            second = np.linalg.norm(features1[i] - features2[1])
            c, s = 0, 1
        else:
            closest = np.linalg.norm(features1[i] - features2[1])
            second = np.linalg.norm(features1[i] - features2[0])
            c, s = 1, 0

        for j in range(2, features2.shape[0]):
            dist = np.linalg.norm(features1[i] - features2[j])
            if dist < second:
                if dist < closest:
                    second = closest
                    closest = dist
                    s = c
                    c = j
                else:
                    second = dist
                    s = j
        raw_match.append([c, s])
        match_dist.append([closest, second])

    valid_match = []
    valid_kp1 = []
    valid_kp2 = []
    for i, m in enumerate(raw_match):
        closest, second = match_dist[i]
        # to eliminate ambiguous matches
        if closest < ratio * second:
            valid_kp1.append(keypoints1[i])
            valid_kp2.append(keypoints2[m[0]])

    return np.asarray(valid_kp1),np.asarray(valid_kp2)

def feature_point_matching(img1,img2,ratio):
    """
    :returns:
        keypoints1: (N,2) ndarray
        keypoints2: (N,2) ndarray
    """
    keypoints1,features1=img2keypointsandfeature(img1)
    keypoints2,features2=img2keypointsandfeature(img2)
    return valid_matching(keypoints1,features1,keypoints2,features2,ratio)