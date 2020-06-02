import numpy as np


def compute_P_from_essential(E):
    """ Compute the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. E = [t]R
    :returns: list of 4 possible camera matrices.
    """
    U, S, V = np.linalg.svd(E)

    # Ensure rotation matrix are right-handed with positive determinant
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    # create 4 possible camera matrices
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    P2s = [np.vstack(((U @ W @ V).T, U[:, 2])).T,
           np.vstack(((U @ W @ V).T,-U[:, 2])).T,
           np.vstack(((U @ W.T @ V).T, U[:, 2])).T,
           np.vstack(((U @ W.T @ V).T,-U[:, 2])).T]

    return P2s

def skew_sym_mat(x):
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ])

def best_P2(P1,P2s,pt1,pt2):
    """
    :param P1: Extrinsic matrix from camera1
    :param P2s: Extrinsic matrixs from camera2
    :param pt1: (3,N) ndarray
    :param pt2: (3,N) ndarray
    :return: [(N,3) ndarray, (N,3) ndarray, (N,3) ndarray, (N,3) ndarray]
    """
    index = -1
    for i, P2 in enumerate(P2s):
        # (pt1 x P1) * X = 0
        # (pt2 x P2) * X = 0
        A = np.vstack((skew_sym_mat(pt1[:, 0]) @ P1,
                       skew_sym_mat(pt2[:, 0]) @ P2))
        U, S, V = np.linalg.svd(A)
        P = np.ravel(V[-1, :4])
        v1 = P / P[3]  # X solution

        P2_h = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        v2 = np.dot(P2_h[:3, :4], v1)

        if v1[2] > 0 and v2[2] > 0:
            index = i

    return P2s[index]

def choose_best_threeD(h1,h2,P1,P2):
    P2 = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))[:3, :4]

    n_point = h1.shape[1]
    res = np.ones((n_point, 3))

    for i in range(n_point):
        A = np.asarray([
            (h1[0, i] * P1[2, :] - P1[0, :]),
            (h1[1, i] * P1[2, :] - P1[1, :]),
            (h2[0, i] * P2[2, :] - P2[0, :]),
            (h2[1, i] * P2[2, :] - P2[1, :])
        ])

        U,S,V = np.linalg.svd(A)
        res[i, :] = V[-1,:-1]/ V[-1,-1]

    return res

