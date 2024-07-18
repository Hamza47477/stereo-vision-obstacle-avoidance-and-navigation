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
from gait_logic.quadruped import Quadruped


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



#------------------------------ Converting depth map to real world 3d coordinates ----------------------------


def calculate_3d_points(disparity_map, Q):
    """ Calculate the 3D point cloud from the disparity map """
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    return points_3d


#-------------------------------- To check the predicted movement of the robot from path ----------------------


def check_movement_direction(points_3d):
    """ Check the first 50 values of the x, y, z axes and determine movement direction """
    r = Quadruped()
    r.calibrate()
    
    
    first_50_points = points_3d[:50]

    x_values = first_50_points[:, 0]
    y_values = first_50_points[:, 1]
    z_values = first_50_points[:, 2]

    if np.all(np.diff(x_values) > 0):
        x_direction = "Move Right"
        r.move(momentum=right_momentum)
        
    elif np.all(np.diff(x_values) < 0):
        x_direction = "Move Left"
        r.move(momentum=left_momentum)
    else:
        x_direction = "No clear direction in X"
        r.move(momentum=forward_momentum)

    if np.all(np.diff(z_values) > 0):
        z_direction = "Move Forward"
        r.move(momentum=forward_momentum)
        
    elif np.all(np.diff(z_values) < 0):
        z_direction = "Move Backward"
        r.move(momentum=backward_momentum)
        
    else:
        z_direction = "No clear direction in Z"
        r.move(momentum=backward_momentum)

    return x_direction, z_direction








if __name__ == '__main__':
    
    
    # Setting the momentum values for movement
    
    right_momentum = np.asarray([0 , 4 , 0 , 0], dtype=np.float32)
    left_momentum = np.asarray([0 , -4 , 0 , 0], dtype=np.float32)
    forward_momentum = np.asarray([4 , 0 , 0 , 0], dtype=np.float32)
    backward_momentum = np.asarray([4 , 0 , 0 , 0], dtype=np.float32)
    
    
    
    #  paths to the images
    left_image_path = r"rectified_images\left_img_rectified4.jpg"
    right_image_path = r"rectified_images\right_img_rectified4.jpg"
    disparity_image_path = r"disparity4.png"



    #  paths to the camera parameters
    camera_matrix_path = r"images copy\calibration_results\CmL.txt"
    distortion_coefficients_path = r"images copy\calibration_results\DcL.txt"



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
    Q = np.array([[1.0, 0.0, 0.0, -776.937],
                    [0.0, 1.0, 0.0, -263.837],
                    [0.0, 0.0, 0.0, 1866.196],
                    [0.0, 0.0, 0.314, 0.0]])




    # Calculate 3D points
    points3d = calculate_3d_points(disparity_image, Q)  # 3-d (1280, 720, 3) x, y, z




    # # Check movement direction based on the first 50 3D points
    # points_3d_flat = points3d.reshape(-1, 3)  # Flatten the 3D points array


    # # Continue with the rest of the original script...
    # points_3d = points3d.reshape(-1, 3)  # converting to 2D array ==> -1 (resulting shape 3-1 = 2), 3 (original shape)

    # colors = cv2.cvtColor(leftFrame, cv2.COLOR_GRAY2RGB).reshape(-1, 3) / 255.0

    # # Applying mask on the disparity map
    # mask = (disparity_image > disparity_image.min())  # defining mask
    # points_3d = points_3d[mask.ravel()]  # Applying mask on points -- ravel() converting it to 1D (flattening)
    # colors = colors[mask.ravel()]  # Applying mask on colors

    # # Creating point cloud object
    # pcd = o3d.geometry.PointCloud()  # initializing
    # pcd.points = o3d.utility.Vector3dVector(points_3d)  # adding vector points to cloud
    # pcd.colors = o3d.utility.Vector3dVector(colors)  # adding color points to cloud

    # # Visualize the point cloud
    # o3d.visualization.draw_geometries([pcd])






    yfloor = 200
    nDisp = 200

    np.set_printoptions(suppress=True, precision=3)                               # setting the printing parameters

    xx, yy, zz = points3d[:, :, 0], points3d[:, :, 1], points3d[:, :, 2]          # extracting the x, y, z values of point cloud -- point cloud = (width, height, x, y, z)
    xx, yy, zz = np.clip(xx, -25, 60), np.clip(yy, -25, 25), np.clip(zz, 0, 100)  # clip values to set limits




    ''' ------------------------Filter obstacles above ground/floor plane --------------------------'''
    
    obs = zz[yfloor - 5:yfloor, :]  
    
    

    '''------------------- Construct occupancy grid----------------------- '''
    
    
    obstacles = np.amin(obs, 0, keepdims=False) 
    y = np.mgrid[0:np.amax(obstacles), 0:obs.shape[1]][0, :, :]  # mesh grid for collection of coordinates -- furthest distance


    ### Assign weights to regions (cost low -> high == 0.01 -> 2)
    occupancy_grid = np.where(y >= obstacles, 0, 1)
    occupancy_grid[:, :nDisp + 50] = 0  # Masking out left side of region
    occupancy_grid[:, -nDisp - 50:] = 0  # Masking out right side of region
    

    # Finding the farthest point
    far_zy, far_zx = np.unravel_index(np.argmax(np.flip(occupancy_grid[:, :-90])), occupancy_grid[:, :-90].shape)  # finding furthest visible point
    far_zx = (zz.shape[1] - 91) - far_zx  # adjust the coordinates back to original
    far_zy = occupancy_grid.shape[0] - far_zy - 1

    xcenter = 640  # Adjusted to the center of the image





    ''' ------------------------A* path-finding config and computation ---------------------------'''
    
    
    mat_grid = Grid(matrix=occupancy_grid)  # creating occupancy grid (environment)
    start = mat_grid.node(xcenter, 1)  # starting position
    end = mat_grid.node(far_zx, far_zy)  # ending position
    tp1 = time.time()
    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)  # path finding
    path, runs = finder.find_path(start, end, mat_grid)  # return shortest path
    print('Raw path :', path)
    tp2 = time.time()
    cost_path = tp2 - tp1

    if len(path) == 0:
        print('ERROR: No path found')
        
        
    
    
    

    ''' ----------------------Map X,Y pixel positions to world-frame for cv.projectPoints() ------------------'''
    
    
    coords = np.array([(xp, zp) for xp, zp in path], dtype=np.int32)

    yrange = np.geomspace(yy.shape[0] - 1, yfloor + 1, num=len(path), dtype=np.int32)
    yrange = np.flip(yrange)

    yworld = np.geomspace(10, 13, num=len(path), dtype=np.float32)
    xworld = xx[yrange, coords[:, 0]]
    zworld = np.array([zp for _, zp in path], dtype=np.float32)
    zworld = np.interp(zworld, [0, np.amax(zworld)], [25, nDisp])

    cf = np.array([xworld, yworld, zworld]).T
    
    
    
    # call the movement function to move
    
    x_direction, z_direction = check_movement_direction(cf)

    print(f"X-Axis Direction: {x_direction}")
    print(f"Z-Axis Direction: {z_direction}")
    

    save_to_txt('3d_coordinates.txt', cf)

    print('world coordinates path :', cf)





    ''' -------------------Reproject 3D world-frame points back to unrectified 2D points------------------'''
    
    pr, _ = cv2.projectPoints(cf, np.zeros(3), np.zeros(3), CL, DL)
    pr = np.squeeze(pr, 1)
    py = pr[:, 1]
    px = pr[:, 0]




    ''' ---------------------------------Draw Floor Polygon -----------------------'''
    
    fPts = np.array([[-40, 13, nDisp], [40, 13, nDisp], [40, 15, 0], [-40, 15, 0]], dtype=np.float32).T
    # fPts order: (top left, top right, bottom right, bottom left)
    pf, _ = cv2.projectPoints(fPts, np.zeros(3).T, np.zeros(3), CL, None)
    pf = np.squeeze(pf, 1)







    '''--------------------------- Update figure (final results)------------------------ '''
    
    
    imL = cv2.imread(r'rectified_images\left_img_rectified4.jpg')
    imL = cv2.cvtColor(imL, cv2.COLOR_BGR2RGB)

    plt.clf()

    pathStats = 'steps={}\npathlen={}'.format(runs, len(path))
    plt.gcf().text(x=0.75, y=0.05, s=pathStats, fontsize='small')

    plt.subplot(231)
    plt.imshow(imL)
    plt.title('Planned Path (Left Camera)')
    plt.xlim([0, 1280])
    plt.ylim([720, 0])
    plt.scatter(px, py, s=np.geomspace(70, 5, len(px)), c=cf[:, 1], cmap=plt.cm.plasma_r, zorder=99)
    plt.gca().add_patch(Polygon(pf, fill=True, facecolor=(0, 1, 0, 0.12), edgecolor=(0, 1, 0, 0.35)))

    ax = plt.gcf().add_subplot(232, projection='3d')
    ax.azim = 90
    ax.elev = 110
    ax.set_box_aspect((4, 3, 3))
    ax.plot_surface(xx[100:yfloor, :], yy[100:yfloor, :], zz[100:yfloor, :], cmap=plt.cm.viridis_r,
                    rcount=25, ccount=25, linewidth=0, antialiased=False)
    ax.set_xlabel('Azimuth (X)')
    ax.set_ylabel('Elevation (Y)')
    ax.set_zlabel('Depth (Z)')
    ax.invert_xaxis()
    ax.invert_zaxis()
    ax.set_title('Planned Path (wrt. world-frame)')
    ax.scatter3D(cf[:, 0], cf[:, 1], cf[:, 2], c=cf[:, 2], cmap=plt.cm.plasma_r)

    plt.subplot(233)
    plt.imshow(occupancy_grid, origin='lower', interpolation='none', cmap='gray')
    plt.title('Occupancy Grid')

    plt.subplot(234)
    plt.imshow(occupancy_grid, origin='lower', interpolation='none', cmap='gray')
    plt.title('Occupancy Grid with A* Path')
    plt.plot(coords[:, 0], coords[:, 1], 'r')  # Plot A* path over occupancy grid

    plt.show()
