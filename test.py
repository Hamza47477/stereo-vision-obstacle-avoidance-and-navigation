import numpy as np
import bezier

class Robot:
    def move(self):
        """
        Walks around based on hardcoded momentum values
        """
        momentum = np.asarray([1, 0.5, 0.25, 0], dtype=np.float32)

        index = 0

        # Generate footstep for front legs
        s_vals = np.linspace(0.0, 1.0, 20)
        front_step_nodes = np.asfortranarray([
            [-1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [-15.0, -10, -10, -15.0],
        ])
        front_curve = bezier.Curve(front_step_nodes, degree=3)
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
        back_curve = bezier.Curve(back_step_nodes, degree=3)
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
            front_trajectory = front_motion * momentum[:3, None]
            back_trajectory = back_motion * momentum[:3, None]
            if momentum[3]:
                close = True
            x_front, z_front, y_front = front_trajectory
            x_back, z_back, y_back = back_trajectory
            
            # Debugging output for front leg positions
            print(f'Index: {index}, Front Leg Positions:')
            print(f'x_front: {x_front}')
            print(f'y_front: {y_front}')
            print(f'z_front: {z_front}')
            
            # Debugging output for back leg positions
            print(f'Index: {index}, Back Leg Positions:')
            print(f'x_back: {x_back}')
            print(f'y_back: {y_back}')
            print(f'z_back: {z_back}')
            
            index += 1
            if index >= 5:  # Limiting the loop to 5 iterations for testing
                close = True

if __name__ == "__main__":
    robot = Robot()
    robot.move()
