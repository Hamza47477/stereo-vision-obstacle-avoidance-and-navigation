import socket
import numpy as np
import argparse
import keyboard

def create_socket_connection(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('0.0.0.0', port))
    print(f"Connection Info - ip: 0.0.0.0, port: {port}")
    return s

def controller(pi_ip, pi_port, accel=4, bound=4, return_to_zero=False):
    s = create_socket_connection(pi_port)
    server = (pi_ip, pi_port)
    momentum = np.array([0., 0., 1., 0.], dtype=np.float32)  # Ensure dtype is float32
    close = False

    while not close:
        moved = False
        if keyboard.is_pressed('w'):
            momentum[0] = min(momentum[0] + accel, bound)
            moved = True
        if keyboard.is_pressed('s'):
            momentum[0] = max(momentum[0] - accel, -bound)
            moved = True
        if keyboard.is_pressed('a'):
            momentum[1] = max(momentum[1] - accel, -bound)
            moved = True
        if keyboard.is_pressed('d'):
            momentum[1] = min(momentum[1] + accel, bound)
            moved = True
        if keyboard.is_pressed('p'):
            momentum[3] = 1
            close = True
            moved = True

        # Not controlling the robot will slowly come to a stop
        if return_to_zero and not moved:
            moved = True
            if momentum[0] > 0:
                momentum[0] -= accel
            elif momentum[0] < 0:
                momentum[0] += accel
            if momentum[1] > 0:
                momentum[1] -= accel
            elif momentum[1] < 0:
                momentum[1] += accel
        
        if moved:
            s.sendto(momentum.tobytes(), server)
            print(momentum)
    
    s.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pi_ip')
    
    parser.add_argument('pi_port', type=int)
    parser.add_argument('--accel', type=float, default=0.002)
    parser.add_argument('--bound', type=int, default=4)
    parser.add_argument('--return_to_zero', action='store_true')
    args = parser.parse_args()

    controller(args.pi_ip, args.pi_port, args.accel, args.bound, args.return_to_zero)
