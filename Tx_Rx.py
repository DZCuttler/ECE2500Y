import socket
import pickle
import time
import threading

HOST = '192.168.2.208'
PORT = 5050


def transmit(data):
    # Serialize
    data = pickle.dumps(data)

    # Send to server

    # Try 10 times to connect to the server
    # If the server is not ready, wait and retry
    for attempt in range(10):
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            break
        except ConnectionRefusedError:
            if attempt == 9:
                raise
            time.sleep(0.001)


    client_socket.sendall(data)
    client_socket.close()


def receive():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)
    conn, addr = server_socket.accept()

    data = b""
    while True:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    conn.close()
    server_socket.close()
    return pickle.loads(data)


def transfer(data):
    # Function to catch and store received data
    received = {}
    def receive_and_store():
        received["data"] = receive()

    # EDGE -> CLOUD
    server_thread = threading.Thread(target=receive_and_store)
    server_thread.start()

    transmit(data)  # transmit

    server_thread.join()

    return received['data']