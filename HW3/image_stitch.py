import numpy as np
import cv2
import os
import sys
import matplotlib.pyplot as plt

def detect_feature_and_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect and extract features from the image
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, features = sift.detectAndCompute(gray, None)

    keypoints = np.float32([i.pt for i in keypoints])
    return keypoints, features

def feature_matching(features1, features2, ratio):
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
        
        raw_match.append((c, s))
        match_dist.append((closest, second))
    
    valid_match = []
    for i, m in enumerate(raw_match):
        (closest, second) = match_dist[i]
        # to eliminate ambiguous matches
        if closest < ratio * second:
            valid_match.append((m[0], i)) 
    
    return valid_match

def drawMatches(image1, image2, keypoints1, keypoints2, matches):
    # combine two images together
    (h1, w1) = image1.shape[:2]
    (h2, w2) = image2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
    vis[0:h1, 0:w1] = image1
    vis[0:h2, w1:] = image2

    # loop over the matches
    for (i, j) in matches:
        # draw the match
        color = np.random.randint(0, high=255, size=(3,)) # make visualization more colorful
        color = tuple([int(x) for x in color])
        pt1 = (int(keypoints1[j][0]), int(keypoints1[j][1]))
        pt2 = (int(keypoints2[i][0]) + w1, int(keypoints2[i][1]))
        cv2.line(vis, pt1, pt2, color, 1)

    return vis

def find_Homography(keypoints1, keypoints2, valid_match, threshold):
    points1 = np.float32([keypoints1[i] for (_,i) in valid_match])
    points2 = np.float32([keypoints2[i] for (i,_) in valid_match])
    #### opencv solution ####
    # (H, status) = cv2.findHomography(points1, points2, cv2.RANSAC, threshold)
    # print(H)
    # inliers = 0
    # for i in status:
    #     if i == 1:
    #         inliers += 1
    # print(inliers)

    length = np.shape(points1)[0]
    mapped_points1 = np.zeros(np.shape(points1), dtype=float)
    original_coord = np.concatenate((points1, np.ones((1,np.shape(points1)[0]), dtype=float).T), axis=1)
    S = 4
    N = 2000
    best_i = 0
    best_H = np.zeros((3,3), dtype=float)

    # RANSAC algorithm to find the best H
    for _ in range(N):
        inliers = 0
        idx = np.random.choice(length, S, replace=False) # sample S index of points
        # compute homography 
        P = np.zeros((S*2,9),np.float32)
        for i in range(S):
            row = i*2
            P[row,:3] = P[row,-3:] = P[row+1,3:6] = P[row+1,-3:] = np.array([points1[idx[i]][0], points1[idx[i]][1], 1.0])
            P[row,-3:] *= -points2[idx[i]][0]
            P[row+1,-3:] *= -points2[idx[i]][1]

        _, _, V = np.linalg.svd(P)
        H = V[-1,:]/V[-1,-1]  # normalize, so H[-1,-1]=1.0
        H = H.reshape((3,3))

        # map points1 from its coordinate to points2 coordinate
        # H @ [x, y, 1].T = lambda * [x', y', 1]
        mapped_coord = original_coord @ H.T
        for i in range(length): 
            mapped_points1[i][0] = mapped_coord[i][0] / mapped_coord[i][2]
            mapped_points1[i][1] = mapped_coord[i][1] / mapped_coord[i][2]
            l = np.linalg.norm(mapped_points1[i] - points2[i])
            if l < threshold:
                inliers += 1
        
        if inliers > best_i:
            best_i = inliers
            best_H = H

    return best_H

def warp(image1, image2, H):
    ##### opencv solution #####
    # width = image1.shape[1] + image2.shape[1]
    # result_image = cv2.warpPerspective(image1, H, (width , image1.shape[0]))
    # result_image[0:image2.shape[0], 0:image2.shape[1]] = image2
    # cv2.imshow("Panorama_image.jpg",result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    h1, w1, h2, w2 = image1.shape[0], image1.shape[1], image2.shape[0], image2.shape[1]
    inv_H = np.linalg.inv(H)
    result_image = np.zeros((h1, w1 + w2, 3),dtype=np.uint8)

    # By hints, there is a problem about continuous and discrete, so don't directly use H to map the image1 to image2
    # In order to get the pixel values, use inv_H to map the coordinate of image2 to image1's 
    for i in range(h2): 
        for j in range(w1 + w2):
            # H @ [x, y, 1].T = lambda * [x', y', 1]
            # inv_H @ [x, y, 1].T = 1/lambda * [x, y, 1]
            coord2 = np.array([j, i, 1])
            coord1 = inv_H @ coord2
            coord1[0] /= coord1[2]
            coord1[1] /= coord1[2]
            coord1 = np.around(coord1[:2])
            new_i, new_j = int(coord1[0]), int(coord1[1]) # find the closest coordinate
            if new_i>=0 and new_j>=0 and new_i<w1 and new_j<h1: # check boundary
                result_image[i][j] = image1[new_j][new_i] # get the pixel values in image1, and map it to the result_image
    
    result_image[0:h2, 0:w2] = image2
    return result_image

def images_stitching(images, ratio, threshold):
    keypoints1, features1 = detect_feature_and_keypoints(images[1]) 
    keypoints2, features2 = detect_feature_and_keypoints(images[0]) 
    valid_match = feature_matching(features1, features2, ratio)
    vis = drawMatches(images[1], images[0], keypoints1, keypoints2, valid_match)
    cv2.imshow('interest_point_match', vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    H = find_Homography(keypoints1, keypoints2, valid_match, threshold)
    result_image = warp(images[1], images[0], H)
    return result_image

if __name__ == '__main__':
    root = os.path.join('data')
    num_img = 6
    ratio = 0.75 # recommend 0.7 to 0.8
    threshold = 4.0 # recommend 0 to 10
    name = []
    name.append('nature1.jpg')
    name.append('nature2.jpg')
    name.append('nature3.jpg')
    name.append('nature4.jpg')
    name.append('nature5.jpg')
    name.append('nature6.jpg')

    images = []
    for i in range(num_img):
        images.append(cv2.imread(os.path.join(root, name[i])))

    if num_img == 2:
        result_image = images_stitching(images, ratio, threshold)
    else:
        result_image = images_stitching([images[-2], images[-1]], ratio, threshold)
        for i in range(num_img - 3, -1, -1):
            result_image = images_stitching([images[i], result_image], ratio, threshold)
    
    cv2.imwrite(os.path.join('result', 'result_nature.jpg'), result_image)
