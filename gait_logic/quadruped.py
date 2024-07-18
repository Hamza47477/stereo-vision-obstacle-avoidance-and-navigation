from adafruit_servokit import ServoKit
from enum import IntEnum
import math
import bezier
import numpy as np

class Motor(IntEnum):
    # identifies the corresponding pin location with the motor location
    FR_SHOULDER = 0
    FR_ELBOW = 1
    FR_HIP = 2
    FL_SHOULDER = 3
    FL_ELBOW = 4
    FL_HIP = 5
    BR_SHOULDER = 6
    BR_ELBOW = 7
    BL_SHOULDER = 8
    BL_ELBOW = 9

class Quadruped:
    def __init__(self):
        self.kit = ServoKit(channels=16)
        self.upper_leg_length = 10
        self.lower_leg_length = 10.5
        for i in range(10):
            self.kit.servo[i].set_pulse_width_range(500, 2500)

    def set_angle(self, motor_id, degrees):
        """
        set the angle of a specific motor to a given angle
        :param motor_id: the motor id
        :param degrees: the angle to put the motor to
        :returns: void
        """
        self.kit.servo[motor_id].angle = degrees

    def rad_to_degree(self, rad):
        """
        Converts radians to degrees
        :param rad: radians
        :returns: the corresponding degrees as a float
        """
        return rad * 180 / math.pi

    def calibrate(self):
        """
        sets the robot into the default "middle position" use this for attaching legs in right location
        :returns: void
        """
        self.set_angle(Motor.FR_SHOULDER, 60)
        self.set_angle(Motor.FR_ELBOW, 90)
        self.set_angle(Motor.FR_HIP, 90)
        self.set_angle(Motor.FL_SHOULDER, 120)
        self.set_angle(Motor.FL_ELBOW, 90)
        self.set_angle(Motor.FL_HIP, 90)
        self.set_angle(Motor.BR_SHOULDER, 60)
        self.set_angle(Motor.BR_ELBOW, 90)
        self.set_angle(Motor.BL_SHOULDER, 120)
        self.set_angle(Motor.BL_ELBOW, 90)

    def inverse_positioning(self, shoulder, elbow, x, y, z=0, hip=None, right=True):
        '''
        Positions the end effector at a given position based on cartesian coordinates in 
        centimeter units and with respect to the shoulder motor
        :param shoulder: motor id used for the shoulder
        :param elbow: motor id used for the elbow
        :param x: cartesian x with respect to shoulder motor (forward/back)
        :param y: cartesian y with respect to shoulder motor (up/down)
        :param z: cartesian z with respect to shoulder motor (left/right)
        :param hip: motor id used for the hip
        :param right: a boolean that flips the logic for left and right side to properly map "forward direction"
        :return: a list containing the appropriate angle for the shoulder and elbow
        '''
        L = 2
        y_prime = -math.sqrt((z + L) ** 2 + y ** 2)
        thetaz = math.atan2(z + L, abs(y)) - math.atan2(L, abs(y_prime))

        elbow_offset = 20
        shoulder_offset = 10
        a1 = self.upper_leg_length
        a2 = self.lower_leg_length

        c2 = (x ** 2 + y_prime ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
        s2 = math.sqrt(1 - c2 ** 2)
        theta2 = math.atan2(s2, c2)
        c2 = math.cos(theta2)
        s2 = math.sin(theta2)

        c1 = (x * (a1 + (a2 * c2)) + y_prime * (a2 * s2)) / (x ** 2 + y_prime ** 2)
        s1 = (y_prime * (a1 + (a2 * c2)) - x * (a2 * s2)) / (x ** 2 + y_prime ** 2)
        theta1 = math.atan2(s1, c1)
        # generate positions with respect to robot motors
        theta_shoulder = -theta1
        theta_elbow = theta_shoulder - theta2
        theta_hip = 0
        if right:
            theta_shoulder = 180 - self.rad_to_degree(theta_shoulder) + shoulder_offset
            theta_elbow = 130 - self.rad_to_degree(theta_elbow) + elbow_offset
            if hip:
                theta_hip = 90 - self.rad_to_degree(thetaz)
        else:
            theta_shoulder = self.rad_to_degree(theta_shoulder) - shoulder_offset
            theta_elbow = 50 + self.rad_to_degree(theta_elbow) - elbow_offset
            if hip:
                theta_hip = 90 + self.rad_to_degree(thetaz)
        self.set_angle(shoulder, theta_shoulder)
        self.set_angle(elbow, theta_elbow)
        if hip:
            self.set_angle(hip, theta_hip)
        # print("theta shoulder:", theta_shoulder, "\ttheta_elbow:", theta_elbow)
        return [theta_shoulder, theta_elbow]

    def leg_position(self, leg_id, x, y, z=0):
        """
        wrapper for inverse position that makes it easier to control each leg for making fixed paths
        :param leg_id: string for the leg to be manipulated
        :param x: cartesian x with respect to shoulder motor (forward/back)
        :param y: cartesian y with respect to shoulder motor (up/down)
        :param z: cartesian z with respect to shoulder motor (left/right)
        """
        if leg_id == 'FL':
            self.inverse_positioning(Motor.FL_SHOULDER, Motor.FL_ELBOW, x, y, z=z, hip=Motor.FL_HIP, right=False)
        if leg_id == 'FR':
            self.inverse_positioning(Motor.FR_SHOULDER, Motor.FR_ELBOW, x, y, z=z, hip=Motor.FR_HIP, right=True)
        if leg_id == 'BL':
            self.inverse_positioning(Motor.BL_SHOULDER, Motor.BL_ELBOW, x, y, right=False)
        if leg_id == 'BR':
            self.inverse_positioning(Motor.BR_SHOULDER, Motor.BR_ELBOW, x, y, right=True)

    def move(self, momentum=None):
        """
        Walks around based on the controller inputted momentum
        :param controller: the controller that is called to determine the robot momentum
        :returns: None, enters an infinite loop 
        """
        
        
        if momentum is None:
            # Default momentum values
            momentum = np.asarray([4, 0, 1, 0], dtype=np.float32)

        """
        The first index (0) represents the momentum or velocity of the robot in the forward/backward direction along the X-axis.
        The second index (1) represents the momentum or velocity of the robot in the left/right direction along the Y-axis.
        The third index (2) represents the momentum or velocity of the robot in the up/down direction along the Z-axis.
        The fourth element (3) is a flag indicating whether the robot should stop or continue moving.
        """
        index = 0

        # Generate footstep for front legs
        s_vals = np.linspace(0.0, 1.0, 20)
        front_step_nodes = np.asfortranarray([
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-15.0, -10, -10, -15.0],
        ])
        front_curve = bezier.Curve(front_step_nodes, degree=3)  # creates a curve object with 3 degree and dimensions
        front_step = front_curve.evaluate_multi(s_vals)
        front_slide_nodes = np.asfortranarray([
            [1.0, -1.0],
            [1.0, -1.0],
            [-15.0, -15],
        ])
        front_slide_curve = bezier.Curve(front_slide_nodes, degree=1)
        front_slide = front_slide_curve.evaluate_multi(s_vals)

        front_motion = np.concatenate((front_step, front_slide), axis=1)

        # Generate footstep for back legs
        back_step_nodes = np.asfortranarray([
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-15.0, -10, -10, -15.0],
        ])
        back_curve = bezier.Curve(back_step_nodes, degree=3)  # creates a curve object with 3 degree and dimensions
        back_step = back_curve.evaluate_multi(s_vals)
        back_slide_nodes = np.asfortranarray([
            [1.0, -1.0],
            [1.0, -1.0],
            [-15.0, -15],
        ])
        back_slide_curve = bezier.Curve(back_slide_nodes, degree=1)
        back_slide = back_slide_curve.evaluate_multi(s_vals)

        back_motion = np.concatenate((back_step, back_slide), axis=1)

        close = False
        while not close:
            momentum = controller(momentum)
            front_trajectory = front_motion * momentum[:3, None]
            back_trajectory = back_motion * momentum[:3, None]
            if momentum[3]:
                close = True
            x_front, z_front, y_front = front_trajectory
            x_back, z_back, y_back = back_trajectory
            # 
            i1 = index % 40
            i2 = (index + 20) % 40 
            # Apply movement based movement
            self.inverse_positioning(Motor.FR_SHOULDER, Motor.FR_ELBOW, x_front[i1], y_front[i1] - 1, z=z_front[i1], hip=Motor.FR_HIP, right=True)
            self.inverse_positioning(Motor.BR_SHOULDER, Motor.BR_ELBOW, x_back[i2], y_back[i2] + 2, right=True)
            self.inverse_positioning(Motor.FL_SHOULDER, Motor.FL_ELBOW, x_front[i2], y_front[i2] - 1, z=-z_front[i2], hip=Motor.FL_HIP, right=False)
            self.inverse_positioning(Motor.BL_SHOULDER, Motor.BL_ELBOW, x_back[i1], y_back[i1] + 2, right=False)
            index += 1
