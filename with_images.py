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



if __name__ == '__main__':
    # Hardcoded paths to the images
    left_image_path = r"rectified_images\left_img_rectified.jpg"
    right_image_path = r"rectified_images\right_img_rectified.jpg"
    disparity_image_path = r"rectified_images\disparity_image.npy"

    # Hardcoded paths to the camera parameters
    camera_matrix_path = r"images\calibration_results\CmL.txt"
    distortion_coefficients_path = r"images\calibration_results\DcL.txt"

    # Load images
    leftFrame = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    rightFrame = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if leftFrame is None or rightFrame is None:
        print(f"Error reading images: {left_image_path}, {right_image_path}")
        sys.exit(-1)

    # Load camera parameters
    CL, DL = load_camera_parameters(camera_matrix_path, distortion_coefficients_path)

    # Compute disparity map
    disparity_image = depth_map(leftFrame, rightFrame)

    # Q matrix (reprojection matrix) using your calibration parameters
    Q = np.array([[1.0, 0.0, 0.0, -646.38246918],
                  [0.0, 1.0, 0.0, -252.22726822],
                  [0.0, 0.0, 0.0, 994.95161541],
                  [0.0, 0.0, 0.300325055, 0.0]])

    # Calculate 3D points
    points3d = calculate_3d_points(disparity_image, Q)

    # Save the results (optional)
    np.save(disparity_image_path, disparity_image)

    # Display results
    cv2.imshow('Disparity Image', disparity_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # For 3D visualization, you can use Open3D or other 3D visualization libraries

    # Create an Open3D point cloud from the points_3d array
    points_3d = points3d.reshape(-1, 3)
    colors = cv2.cvtColor(leftFrame, cv2.COLOR_GRAY2RGB).reshape(-1, 3) / 255.0

    mask = (disparity_image > disparity_image.min())
    points_3d = points_3d[mask.ravel()]
    colors = colors[mask.ravel()]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

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
    imL = cv2.imread(r'rectified_images\left_img_rectified.jpg')
    imL = cv2.cvtColor(imL, cv2.COLOR_BGR2RGB)
    
    plt.clf()
    
    pathStats = 'steps={}\npathlen={}'.format(runs, len(path))
    plt.gcf().text(x=0.75, y=0.05, s=pathStats, fontsize='small')
    
    plt.subplot(231); plt.imshow(imL); plt.title('Planned Path (Left Camera)')
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

    plt.subplot(233); plt.imshow(occupancy_grid, origin='lower', interpolation='none', cmap='gray')
    plt.title('Occupancy Grid')
    
    plt.subplot(234); plt.imshow(occupancy_grid, origin='lower', interpolation='none', cmap='gray')
    plt.title('Occupancy Grid with A* Path')
    plt.plot(coords[:,0], coords[:,1], 'r')     # Plot A* path over occupancy grid

    plt.show()
