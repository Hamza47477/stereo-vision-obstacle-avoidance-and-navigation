import numpy as np
import cv2
import sys
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import time

#---------------------------- Loading the camera parameters -----------------
def load_camera_parameters(camera_matrix_path, distortion_coefficients_path):
    """ Load camera matrix and distortion coefficients from files. """
    CL = np.loadtxt(camera_matrix_path)
    DL = np.loadtxt(distortion_coefficients_path)
    return CL, DL

#---------------------------- Save the 3d points for path -----------------
def save_to_txt(filename, data):
    """ Save data to a text file """
    with open(filename, 'w') as f:
        for item in data:
            f.write(f"{item[0]} {item[1]} {item[2]}\n")

#------------------------------ Depth map calculation ----------------------
def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map (left to right disparity) """
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=5 * 16,  # max_disp has to be divisible by 16 e.g. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    return filteredImg

#------------------------------ Converting depth map to real world 3d coordinates ----------------------------
def calculate_3d_points(disparity_map, Q):
    """ Calculate the 3D point cloud from the disparity map """
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3d

#-------------------------------- To check the predicted movement of the robot from path ----------------------
def check_movement_direction(points_3d):
     #Check the first 50 values of the x, y, z axes and determine movement direction 
    first_50_points = points_3d[:10]

    x_values = first_50_points[:, 0]
    y_values = first_50_points[:, 1]
    z_values = first_50_points[:, 2]

    if np.all(np.diff(x_values) > 0):
        x_direction = "Move Right"
    elif np.all(np.diff(x_values) < 0):
        x_direction = "Move Left"
    else:
        x_direction = "No clear direction in X"

    if np.all(np.diff(z_values) > 0):
        z_direction = "Move Forward"
    elif np.all(np.diff(z_values) < 0):
        z_direction = "Move Backward"
    else:
        z_direction = "No clear direction in Z"

    return x_direction, z_direction

if __name__ == '__main__':
    # Setting the momentum values for movement
    right_momentum = np.asarray([0 , 4 , 0 , 0], dtype=np.float32)
    left_momentum = np.asarray([0 , -4 , 0 , 0], dtype=np.float32)
    forward_momentum = np.asarray([4 , 0 , 0 , 0], dtype=np.float32)
    backward_momentum = np.asarray([4 , 0 , 0 , 0], dtype=np.float32)

    # Paths to the camera parameters
    camera_matrix_path = r"images copy\calibration_results\CmL.txt"
    distortion_coefficients_path = r"images copy\calibration_results\DcL.txt"

    # Load camera parameters
    CL, DL = load_camera_parameters(camera_matrix_path, distortion_coefficients_path)

    # Initialize the video capture for the left and right cameras
    capL = cv2.VideoCapture(0)
    capR = cv2.VideoCapture(1)

    if not capL.isOpened() or not capR.isOpened():
        print("Error: Could not open video capture")
        sys.exit(-1)

    # Q matrix (reprojection matrix) using your calibration parameters
    Q = np.array([[1.0, 0.0, 0.0, -646.38],
                  [0.0, 1.0, 0.0, -252.22],
                  [0.0, 0.0, 0.0, 994.95],
                  [0.0, 0.0, 0.3003, 0.0]])

    while True:
        # Capture frames from both cameras
        retL, leftFrame = capL.read()
        retR, rightFrame = capR.read()

        if not retL or not retR:
            print("Error: Could not read frames from the cameras")
            break

        # Convert frames to grayscale
        leftGray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
        rightGray = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

        # Compute disparity map
        disparity_image = depth_map(leftGray, rightGray)

        # Calculate 3D points
        points3d = calculate_3d_points(disparity_image, Q)

        yfloor = 100
        nDisp = 100

        np.set_printoptions(suppress=True, precision=3)

        xx, yy, zz = points3d[:, :, 0], points3d[:, :, 1], points3d[:, :, 2]
        xx, yy, zz = np.clip(xx, -25, 60), np.clip(yy, -25, 25), np.clip(zz, 0, 100)

        ''' ------------------------Filter obstacles above ground/floor plane --------------------------'''
        obs = zz[yfloor - 5:yfloor, :]  

        '''------------------- Construct occupancy grid----------------------- '''
        obstacles = np.amin(obs, 0, keepdims=False) 
        y = np.mgrid[0:np.amax(obstacles), 0:obs.shape[1]][0, :, :] 

        ### Assign weights to regions (cost low -> high == 0.01 -> 2)
        occupancy_grid = np.where(y >= obstacles, 0, 1)
        occupancy_grid[:, :nDisp + 50] = 0
        occupancy_grid[:, -nDisp - 50:] = 0

        # Finding the farthest point
        far_zy, far_zx = np.unravel_index(np.argmax(np.flip(occupancy_grid[:, :-90])), occupancy_grid[:, :-90].shape)
        far_zx = (zz.shape[1] - 91) - far_zx
        far_zy = occupancy_grid.shape[0] - far_zy - 1

        xcenter = 640

        ''' ------------------------A* path-finding config and computation ---------------------------'''
        mat_grid = Grid(matrix=occupancy_grid)
        start = mat_grid.node(xcenter, 1)
        end = mat_grid.node(far_zx, far_zy)
        tp1 = time.time()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, mat_grid)
        print('Raw path :', path)
        tp2 = time.time()
        cost_path = tp2 - tp1

        if len(path) == 0:
            print('ERROR: No path found')
            continue

        ''' ----------------------Map X,Y pixel positions to world-frame for cv.projectPoints() ------------------'''
        coords = np.array([(xp, zp) for xp, zp in path], dtype=np.int32)
        yrange = np.geomspace(yy.shape[0] - 1, yfloor + 1, num=len(path), dtype=np.int32)
        yrange = np.flip(yrange)
        yworld = np.geomspace(10, 13, num=len(path), dtype=np.float32)
        coords = np.column_stack((coords, yworld))
        save_to_txt("robot_path.txt", coords)
        print(coords.shape)

        x_dir, z_dir = check_movement_direction(coords)
        print("X Direction:", x_dir)
        print("Z Direction:", z_dir)

        # Display the disparity map
        cv2.imshow("Disparity", disparity_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    capL.release()
    capR.release()
    cv2.destroyAllWindows()
