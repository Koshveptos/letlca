import socket
import json
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

class DroneClient:
    def __init__(self, ip='127.0.0.1', send_port = 3800, recv_port = 3801):
        self.ip = ip
        self.send_port = send_port
        self.recv_port = recv_port
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(('', recv_port))
            self.sock.settimeout(0.05)
        except :
            raise None
        
    def send(self, vx, vy):
         msg = json.dumps({"Vx": float(vx), "Vy": float(vy)})
         self.sock.sendto(msg.encode('utf-8'), (self.ip, self.send_port))

    def receive(self):
        try:
            data, _ = self.sock.recvfrom(1024)
            return json.loads(data.decode('utf-8'))
        except :
            raise None
        
    def one_step(self, vx, vy):
        self.send(vx,vy)
        time.sleep(0.01)
        return self.receive()
    def close_connection(self):
        self.sock.close()
        

        

    
