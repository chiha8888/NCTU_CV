import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)


#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print('Camera calibration...')
img_size = (img.shape[1], img.shape[0])
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
'''if using opencv function:
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
Vr = np.array(rvecs)
Tr = np.array(tvecs)
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)
'''

"""
get homography H_list from all images
solve P@m=0
    P: (2*#image,9) ndarray
    m: (9,1) ndarray
"""
H_list=[]   # 10*(3*3)
for pic in range(len(images)):
    P=np.zeros((corner_x*corner_y*2,9),np.float32)
    for i in range(corner_x*corner_y):
        row=i*2
        P[row,:3]=P[row,-3:]=P[row+1,3:6]=P[row+1,-3:]=np.append(objpoints[pic][i,:-1],1.)
        P[row,-3:]*=-imgpoints[pic][i,0,0]
        P[row+1,-3:]*=-imgpoints[pic][i,0,1]
    _,_,V=np.linalg.svd(P)
    H=V[-1,:]/V[-1,-1]  # normalize, so H[-1,-1]=1.0
    H=H.reshape((3,3))
    H_list.append(H)

"""
get B from H
solve V@b=0
    V: (2*#image,6) ndarray
    b: (6,1) ndarray 
"""
V=np.zeros((len(images)*2,6),np.float32)
for pic in range(len(images)):
    H=H_list[pic]
    V[pic*2,:]=np.array([H[0,0]*H[0,1],
                         H[0,0]*H[1,1]+H[0,1]*H[1,0],
                         H[0,0]*H[2,1]+H[2,0]*H[0,1],
                         H[1,0]*H[1,1],
                         H[1,0]*H[2,1]+H[2,0]*H[1,1],
                         H[2,0]*H[2,1]],np.float32)
    V[pic*2+1,:]=np.array([H[0,0]**2-H[0,1]**2,
                           2*(H[0,0]*H[1,0]-H[0,1]*H[1,1]),
                           2*(H[0,0]*H[2,0]-H[0,1]*H[2,1]),
                           H[1,0]**2-H[1,1]**2,
                           2*(H[1,0]*H[2,0]-H[1,1]*H[2,1]),
                           H[2,0]**2-H[2,1]**2
                           ],np.float32)
_,_,V=np.linalg.svd(V)
B=V[-1,:]
B=np.array([[B[0],B[1],B[2]],
            [B[1],B[3],B[4]],
            [B[2],B[4],B[5]]],np.float32)
# turn B into positive definite
if B[0,0]<0:
    B=-B

"""
get K from B
"""
K_inv=np.linalg.cholesky(B).T
K=np.linalg.inv(K_inv)
K/=K[-1,-1] # normalize, so K[-1,-1]=1.0

"""
get r1,r2,t from K
"""
extrinsics=np.zeros((len(images),6))
for i in range(len(H_list)):
    H=H_list[i]
    _lambda=1/np.linalg.norm(K_inv@H[:,0:1])
    ex=_lambda*K_inv@H
    r1=ex[:,0:1]
    r2=ex[:,1:2]
    r3=np.cross(r1.T,r2.T).T

    R=np.hstack((r1, r2, r3))
    t=ex[:,2:3]
    rot_vec,_=cv2.Rodrigues(R)
    extrinsics[i,:]=np.concatenate((rot_vec,t)).reshape(-1)


# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
ax = fig.gca(projection='3d')
# camera setting
camera_matrix = K
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                scale_focal, extrinsics, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
plt.show()
plt.pause(1)
#animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
