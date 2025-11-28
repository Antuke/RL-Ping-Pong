from client import Client
import time

port = 333
host = "localhost"
name = "serve_trainer"


if __name__ == "__main__":
    client = Client(name, host, port=port)
    print("Connected to server")
    j = client.get_state()[0:11]
    while True:
        j[1] = -100*client.get_state()[17]
        client.send_joints(j)
        ball_pos = client.get_state()[17:20]
        time.sleep(0.2)