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

def threeD_from_camera_matrix(K,P1,P2s,inlier1,inlier2):
    """
    :param K: Intrinsic matrix
    :param P1: Extrinsic matrix from camera1
    :param P2s: Extrinsic matrixs from camera2
    :param inlier1: (N,2) ndarray
    :param inlier2: (N,2) ndarray
    :return: [(N,3) ndarray, (N,3) ndarray, (N,3) ndarray, (N,3) ndarray]
    """
    N=len(inlier1)
    threeDs=[]
    for i in range(4):
        P2=P2s[i]
        CM1=K@P1
        CM2=K@P2

        threeD_points=[]
        for i in range(N):
            u,v=inlier1[i]
            u_,v_=inlier2[i]
            A = np.asarray([u*CM1[2,:]-CM1[0,:],
                            v*CM1[2,:]-CM1[1,:],
                            u_*CM2[2,:]-CM2[0,:],
                            v_*CM2[2,:]-CM2[1,:]])
            U,S,V=np.linalg.svd(A)
            threeD_points.append(V[-1,:-1]/V[-1,-1])
        threeDs.append(np.asarray(threeD_points))

    return threeDs