import numpy as np
import cv2
import time
import os

# Set the camera index
camera_index = 0

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

# Create the directory to save images if it doesn't exist
output_dir = 'images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

        # Display the images
        cv2.imshow('Left Image', left_image)
        cv2.imshow('Right Image', right_image)

        # Wait for 1 millisecond for a key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the images when 's' is pressed
            timestamp = int(time.time())
            left_image_filename = os.path.join(output_dir, f'left_image_{timestamp}.png')
            right_image_filename = os.path.join(output_dir, f'right_image_{timestamp}.png')
            cv2.imwrite(left_image_filename, left_image)
            cv2.imwrite(right_image_filename, right_image)
            print(f"Saved {left_image_filename} and {right_image_filename}")
    else:
        print("Error: Failed to capture frame.")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
