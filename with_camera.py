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
import pickle

def load_camera_parameters(camera_matrix_path, distortion_coefficients_path):
    """ Load camera matrix and distortion coefficients from files. """
    CL = np.loadtxt(camera_matrix_path)
    DL = np.loadtxt(distortion_coefficients_path)
    return CL, DL

def depth_map(imgL, imgR):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map (left to right disparity) """
    # SGBM Parameters -----------------
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

def calculate_3d_points(disparity_map, Q):
    """ Calculate the 3D point cloud from the disparity map """
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3d

# Load calibration data from a .pkl file
with open(r"images\calibration_results\calibration_data.pkl", "rb") as f:
    calibration_data = pickle.load(f)

# Extract calibration parameters
left_camera_matrix = np.array(calibration_data["M1"])
left_dist_coeffs = np.array(calibration_data["dist1"])
right_camera_matrix = np.array(calibration_data["M2"])
right_dist_coeffs = np.array(calibration_data["dist2"])
R = np.array(calibration_data["R"])  # Rotation matrix
T = np.array(calibration_data["T"])  # Translation vector

# Function to rectify stereo images
def rectify_stereo_images(left_img, right_img):
    # Image size
    image_size = left_img.shape[::-1]

    # Stereo rectification
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(left_camera_matrix, left_dist_coeffs,
                                                right_camera_matrix, right_dist_coeffs,
                                                image_size, R, T)

    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_16SC2)

    # Rectify images
    left_img_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_img_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

    return left_img_rectified, right_img_rectified

# Set the camera index
camera_index = 1

# Desired preview resolution (try a lower resolution if necessary)
preview_width = 2560
preview_height = 720

# Open the camera
cap = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera at index", camera_index)
    exit()

# Attempt to set the preview resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)

# Check if the resolution was set correctly
current_preview_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
current_preview_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if current_preview_width != preview_width or current_preview_height != preview_height:
    print(f"Error: Failed to set camera at index {camera_index} to the specified preview format.")
    print(f"Current preview resolution: {current_preview_width}x{current_preview_height}")
    cap.release()
    exit()
else:
    print(f"Camera at index {camera_index} configured successfully with preview format: {preview_width}x{preview_height}")

# Display live video stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is captured successfully
    if ret:
        # Split the frame into left and right images
        width = frame.shape[1]
        left_image = frame[:, :width // 2, :]
        right_image = frame[:, width // 2:, :]

        # Convert images to grayscale for depth map calculation
        left_image_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_image_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        # Rectify stereo images
        left_image_rectified, right_image_rectified = rectify_stereo_images(left_image_gray, right_image_gray)

        # Display rectified images
        # cv2.imshow('Rectified Left Image', left_image_rectified)
        # cv2.imshow('Rectified Right Image', right_image_rectified)
        
        # Hardcoded paths to the camera parameters
        camera_matrix_path = r"images\calibration_results\CmL.txt"
        distortion_coefficients_path = r"images\calibration_results\DcL.txt"

        # Load camera parameters
        CL, DL = load_camera_parameters(camera_matrix_path, distortion_coefficients_path)

        # Compute disparity map
        disparity_image = depth_map(left_image_gray, right_image_gray)

        # Q matrix (reprojection matrix) using your calibration parameters
        Q = np.array([[1.0, 0.0, 0.0, -646.38246918],
                      [0.0, 1.0, 0.0, -252.22726822],
                      [0.0, 0.0, 0.0, 994.95161541],
                      [0.0, 0.0, 0.300325055, 0.0]])

        # Calculate 3D points
        points3d = calculate_3d_points(disparity_image, Q)

        yfloor = 100
        nDisp = 100

        np.set_printoptions(suppress=True, precision=3)
        xx, yy, zz = points3d[:,:,0], points3d[:,:,1], points3d[:,:,2]
        xx, yy, zz = np.clip(xx, -25, 60), np.clip(yy, -25, 25), np.clip(zz, 0, 100)

        ''' Filter obstacles above ground/floor plane '''
        obs = zz[yfloor-10:yfloor,:]

        ''' Construct occupancy grid '''
        obstacles = np.amin(obs, 0, keepdims=False)
        y = np.mgrid[0:np.amax(obstacles), 0:obs.shape[1]][0,:,:]

        ### Assign weights to regions (cost low -> high == 0.01 -> 2)
        occupancy_grid = np.where(y >= obstacles, 0, 1)
        occupancy_grid[:, :nDisp+60] = 0

        far_zy, far_zx = np.unravel_index(np.argmax(np.flip(occupancy_grid[:,:-90])), occupancy_grid[:,:-90].shape)
        far_zx = (zz.shape[1]-91) - far_zx
        far_zy = occupancy_grid.shape[0] - far_zy - 1

        xcenter = 800

        ''' A* path-finding config and computation '''
        mat_grid = Grid(matrix=occupancy_grid)
        start = mat_grid.node(xcenter, 1)
        end = mat_grid.node(far_zx, far_zy)
        tp1 = time.time()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, mat_grid)
        tp2 = time.time()
        cost_path = tp2-tp1

        if len(path) == 0:
            print('ERROR: No path found')

        ''' Map X,Y pixel positions to world-frame for cv.projectPoints() '''
        coords = np.array([(xp, zp) for xp, zp in path], dtype=np.int32)

        yrange = np.geomspace(yy.shape[0]-1, yfloor+1, num=len(path), dtype=np.int32)
        yrange = np.flip(yrange)

        yworld = np.geomspace(10,13, num=len(path), dtype=np.float32)
        xworld = xx[yrange, coords[:,0]]
        zworld = np.array([zp for _, zp in path], dtype=np.float32)
        zworld = np.interp(zworld, [0, np.amax(zworld)], [25, nDisp])

        cf = np.array([xworld, yworld, zworld]).T

        ''' Reproject 3D world-frame points back to unrectified 2D points'''
        pr, _ = cv2.projectPoints(cf, np.zeros(3), np.zeros(3), CL, DL)
        pr = np.squeeze(pr, 1)
        py = pr[:,1]
        px = pr[:,0]

        ''' Draw Floor Polygon '''
        fPts = np.array([[-40, 13, nDisp], [40, 13, nDisp], [40, 15, 0], [-40, 15, 0]], dtype=np.float32).T
        # fPts order: (top left, top right, bottom right, bottom left)
        pf, _ = cv2.projectPoints(fPts, np.zeros(3).T, np.zeros(3), CL, None)
        pf = np.squeeze(pf, 1)

        ''' Update figure (final results) '''
        plt.clf()

        pathStats = 'steps={}\npathlen={}'.format(runs, len(path))
        plt.gcf().text(x=0.75, y=0.05, s=pathStats, fontsize='small')

        plt.subplot(231); plt.imshow(left_image_gray, cmap='gray'); plt.title('Planned Path (Left Camera)')
        plt.xlim([0, 1280]); plt.ylim([720, 0])
        plt.scatter(px, py, s=np.geomspace(70, 5, len(px)), c=cf[:,1], cmap=plt.cm.plasma_r, zorder=99)
        plt.gca().add_patch(Polygon(pf, fill=True, facecolor=(0,1,0,0.12), edgecolor=(0,1,0,0.35)))

        ax = plt.gcf().add_subplot(232, projection='3d')
        ax.azim = 90; ax.elev = 110; ax.set_box_aspect((4,3,3))
        ax.plot_surface(xx[100:yfloor,:], yy[100:yfloor,:], zz[100:yfloor,:], cmap=plt.cm.viridis_r, 
                                    rcount=25, ccount=25, linewidth=0, antialiased=False)
        ax.set_xlabel('Azimuth (X)'); ax.set_ylabel('Elevation (Y)'); ax.set_zlabel('Depth (Z)')
        ax.invert_xaxis(); ax.invert_zaxis(); ax.set_title('Planned Path (wrt. world-frame)')
        ax.scatter3D(cf[:,0], cf[:,1], cf[:,2], c=cf[:,2], cmap=plt.cm.plasma_r)

        # plt.subplot(233); plt.imshow(occupancy_grid, origin='lower', interpolation='none', cmap='gray')
        # plt.title('Occupancy Grid')

        plt.subplot(234); plt.imshow(occupancy_grid, origin='lower', interpolation='none', cmap='gray')
        plt.title('Occupancy Grid with A* Path')
        plt.plot(coords[:,0], coords[:,1], 'r')     # Plot A* path over occupancy grid

        plt.show()

        # Wait for 1 millisecond for a key press; exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Failed to capture frame.")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
