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

def threeD_from_camera_matrix(K1,K2,P1,P2s,inlier1,inlier2):
    """
    :param K1: Intrinsic matrix form camera1
    :param K2: Intrinsic matrix from camera2
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
        CM1=K1@P1
        CM2=K2@P2

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

def choose_best_threeD(threeDs,P2s):
    best_front_num=0
    best_idx=-1
    for i in range(4):
        P2=P2s[i]
        R=P2[:,:-1]
        t=P2[:,-1:]
        threeD=threeDs[i].T
        camera_center=-R.T@t
        front_num = np.sum((R[-1:, :]@(threeD - camera_center)) > 0)
        if front_num>best_front_num:
            best_front_num=front_num
            best_idx=i
        print(f'# front num: {front_num}/{threeD.shape[1]}')

    return threeDs[best_idx]

