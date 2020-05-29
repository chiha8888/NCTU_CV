import numpy as np


def normalize_coordinate(points):
    """ Scale and translate image points so that centroid of the points
        are at the origin and avg distance to the origin is equal to sqrt(2).
    :param points: (3,8) ndarray
    """
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    T = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0,     0,                  1]
        ])
    return T, T@points

def compute_fundamental_matrix(x,x_):
    """
    :param x: (3,8) ndarray
    :param x_: (3,8) ndarray
    """
    #Each row in the A is [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    A = np.zeros((8,9))
    for i in range(8):

        A[i] = [x[0, i] * x_[0, i],  x[0, i] * x_[1, i],  x[0, i],
                x[1, i] * x_[0, i],  x[1, i] * x_[1, i],  x[1, i],
                          x_[0, i],            x_[1, i],        1,
                ]
        #A[i]=[ x_[0, i]*x[0, i], x_[0, i]*x[1, i], x_[0, i], x_[1, i]*x[0, i], x_[1, i]*x[1, i], x_[1, i], x[0, i], x[1, i], 1 ]

    # A@f=0
    U, S, V = np.linalg.svd(A)
    F = V[-1,:].reshape(3, 3)

    # constrain F. Make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V
    return F / F[-1,-1]

def compute_fundamental_matrix_normalized(p1,p2):
    """
    :param p1: (8,3) ndarray
    :param p2: (8,3) ndarray
    """
    # preprocess image coordinates
    T1,p1_normalized = normalize_coordinate(p1.T)
    T2,p2_normalized = normalize_coordinate(p2.T)

    F = compute_fundamental_matrix(p1_normalized,p2_normalized)

    # reverse preprocessing of coordinates
    F = T1.T @ F @ T2
    return F / F[2, 2]

def get_fundamental_matrix(keypoints1,keypoints2,threshold):
    """
    :param keypoints1: (N,2) ndarray
    :param keypoints2: (N,2) ndarray
    :param threshold: |x'Fx| < threshold as inliers
    """
    N=len(keypoints1)
    keypoints1=np.hstack((keypoints1,np.ones((N,1))))
    keypoints2=np.hstack((keypoints2,np.ones((N,1))))

    best_cost=1e9
    best_F=None
    best_inlier_idxs=None
    # find best F with RANSAC
    for _ in range(2000):
        choose_idx=np.random.choice(N, 8, replace=False)  # sample 8 correspondence feature points
        # get F
        F=compute_fundamental_matrix_normalized(keypoints1[choose_idx,:],keypoints2[choose_idx,:])

        # select indices with accepted points, Sampson distance as error.
        Fx1 = F @ keypoints1.T
        Fx2 = F @ keypoints2.T
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        errors = np.diag( keypoints1 @ F @ keypoints2.T ) ** 2 / denom
        inlier_idxs=np.where(errors<threshold)[0]

        cost = np.sum(errors[errors<threshold]) + (N-len(inlier_idxs))*threshold
        if cost < best_cost:
            best_cost=cost
            best_F=F
            best_inlier_idxs=inlier_idxs

    return best_F, best_inlier_idxs