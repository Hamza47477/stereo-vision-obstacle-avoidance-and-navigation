import cv2
import numpy as np
import matplotlib.pyplot as plt

def depth_map(imgL_path, imgR_path, output_path, disparity_before_filter_path):
    """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map (left to right disparity) """
    # Read the images
    imgL = cv2.imread(imgL_path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(imgR_path, cv2.IMREAD_GRAYSCALE)

    if imgL is None or imgR is None:
        raise ValueError("Error: One or both images not found or cannot be read")

    # SGBM Parameters -----------------
    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

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

    # Compute the disparity maps
    displ = left_matcher.compute(imgL, imgR)
    dispr = right_matcher.compute(imgR, imgL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)

    # Normalize the disparity map before filtering for display
    displ_normalized = cv2.normalize(src=displ, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    displ_normalized = np.uint8(displ_normalized)

    # Save the disparity map before filtering
    cv2.imwrite(disparity_before_filter_path, displ_normalized)
    print(f"Disparity map before filtering saved to {disparity_before_filter_path}")

    # Display the disparity map before filtering
    plt.imshow(displ_normalized, cmap='gray')
    plt.title('Disparity Map Before Filtering')
    plt.show()

    # Apply the WLS filter
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    # Save the resulting image
    cv2.imwrite(output_path, filteredImg)
    print(f"Depth map saved to {output_path}")

# Example usage
depth_map(r'rectified_images\left_img_rectified4.jpg', r'rectified_images\right_img_rectified4.jpg', r'disparity4.png', r'disparity_before_filter4.png')
