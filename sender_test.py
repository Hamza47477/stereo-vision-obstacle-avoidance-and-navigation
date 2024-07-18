import socket
import argparse

def udp_sender(ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (ip, port)

    try:
        while True:
            message = input("Enter message to send (type 'exit' to quit): ")
            if message.lower() == 'exit':
                break
            s.sendto(message.encode('utf-8'), server_address)
            print(f"Sent message: {message}")
    finally:
        s.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', help="Receiver's IP address")
    parser.add_argument('port', type=int, help="Receiver's port")
    args = parser.parse_args()

    udp_sender(args.ip, args.port)
