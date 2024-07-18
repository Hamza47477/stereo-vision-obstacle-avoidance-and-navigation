import pickle
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

# Load stereo images
left_img = cv2.imread(r"left_image_1719902890.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(r"right_image_1719902890.png", cv2.IMREAD_GRAYSCALE)

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

# Create directory to save rectified images if it doesn't exist
output_dir = r"rectified_images"
os.makedirs(output_dir, exist_ok=True)

# Save rectified images
cv2.imwrite(os.path.join(output_dir, 'left_img_rectified5.jpg'), left_img_rectified)
cv2.imwrite(os.path.join(output_dir, 'right_img_rectified5.jpg'), right_img_rectified)

# Optionally, display the rectified images
cv2.imshow('left_img_rectified', left_img_rectified)
cv2.imshow('right_img_rectified', right_img_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract x and y maps
left_map1_x, left_map1_y = left_map1[:,:,0], left_map1[:,:,1]
right_map1_x, right_map1_y = right_map1[:,:,0], right_map1[:,:,1]

# Plot rectification maps
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(left_map1_x, cmap='gray')
plt.title('Left Map 1 - X coordinates')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(left_map1_y, cmap='gray')
plt.title('Left Map 1 - Y coordinates')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(right_map1_x, cmap='gray')
plt.title('Right Map 1 - X coordinates')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(right_map1_y, cmap='gray')
plt.title('Right Map 1 - Y coordinates')
plt.colorbar()

plt.tight_layout()
plt.show()
