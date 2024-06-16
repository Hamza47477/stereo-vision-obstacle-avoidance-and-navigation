import pickle
import cv2
import numpy as np
import os

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
left_img = cv2.imread(r"Dataset1\left\left_12.png", cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(r"Dataset1\right\right_12.png", cv2.IMREAD_GRAYSCALE)

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
cv2.imwrite(os.path.join(output_dir, 'left_img_rectified2.jpg'), left_img_rectified)
cv2.imwrite(os.path.join(output_dir, 'right_img_rectified2.jpg'), right_img_rectified)

# Optionally, display the rectified images
cv2.imshow('left_img_rectified', left_img_rectified)
cv2.imshow('right_img_rectified', right_img_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
