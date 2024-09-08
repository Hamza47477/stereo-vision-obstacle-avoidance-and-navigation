<<<<<<< HEAD
# quadruped-robot
![quadruped](https://github.com/Hamza47477/Quadruped-robot-inverse-kinematics/blob/main/Robot_hardware.png)


## Installation
To use this code simply download the repository onto the raspberry pi, install the requirements, set up the pi for camera and the servo library usage, and run `python3 control-quadruped`. This will start the walking motion and print out the pi's IP and port. Run the controller on your computer, setting the IP and port given by the pi and the robot will start taking momentum data from the controller.

### Computer vision controller
The computer vision controller has a few extra steps that can seen here.
Specific to the computer-vision controller you will also need to run `python3 image-sender/rpi_send_video.py` to pass camera data to you computer, in this case make sure to change the IP and Port to the ones used by the controller.

### Stereovisioon Navigation 
It involves the calibration of stereo camera and finding parameters for rectification and projection matrix. Then stereo camera calculates the disparity map which is then reprojected to #d point cloud. Yfloor values with certain height will marked as obstacles and A* algorithm is used to find the path avoiding the obstacles.
![quadruped](https://github.com/Hamza47477/stereo-vision-obstacle-avoidance-and-navigation/blob/main/results/1.png)



### Robot Control
The robot uses an inverse kinematic model to determine how to position the foot in the requested location. Some of the math for this can be seen in the model directory with the jupyter notebook 
=======

https://github.com/Hamza47477/stereo-vision-obstacle-avoidance-and-navigation/blob/main/results/1.png
>>>>>>> e3af667e6c61130686331d0f7ef31ed86d8ff406
