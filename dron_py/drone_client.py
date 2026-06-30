import socket
import json
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from config import MAX_DISTANCE, MAX_SPEED, MAX_STEPS, MODEL_SAVE_PATH, REACH_RADIUS, TARGET_X, TARGET_Y, TOTAL_TIMESTEPS

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
            return  None
        
    def send(self, vx, vy):
         msg = json.dumps({"Vx": float(vx), "Vy": float(vy)})
         self.sock.sendto(msg.encode('utf-8'), (self.ip, self.send_port))

    def receive(self):
        if not self.sock:
            return None
        try:
            data, _ = self.sock.recvfrom(1024)
            return json.loads(data.decode('utf-8'))
        except :
            return  None
        
    def one_step(self, vx, vy):
        self.send(vx,vy)
        time.sleep(0.01)
        return self.receive()
    def close_connection(self):
        if self.sock:
            self.sock.close()
        
    
class DroneEnv(gym.Env):
    def __init__(self, mode='virtual'):
        super().__init__()
        self.mode = mode
        self.drone = DroneClient() if mode == 'ue5' else None

        self.target = np.array([TARGET_X, TARGET_Y])
        self.pos = np.array([0.0 , 0.0])
        self.steps = 0
        self.observation_space = spaces.Box(-1000, 1000, (4,), dtype=np.float32)
        self.action_space = spaces.Box(-MAX_SPEED, MAX_SPEED, (2,), dtype=np.float32)
    def reset(self, seed = None, options=None):
        super().reset(seed = seed)
        self.pos = np.array([0.0, 0.0])
        self.steps = 0
        if self.mode == 'ue5' and self.drone:
            self.drone.step(0,0)
        return self._get_obs(), {}
    def step(self, action):
        vx, vy = np.clip(action, -MAX_SPEED, MAX_SPEED)
        #virtual
        if self.mode == 'virtual':
            dt = 0.1
            self.pos  += np.array([vx, vy]) * dt
            self.pos = np.clip(self.pos, -1000, 1000)
        # if reaaaalllll
        else:
            if self.drone:
                state = self.drone.step(vx,  vy)
                if state:
                    self.pos = np.array([state.get('X', 0), state.get('Y',0)])
        self.steps += 1

        dist = np.linalg.norm(self.pos - self.target)
        reward = -dist / 100
        done = False
        if dist < REACH_RADIUS:
            reward += 100
            done = True
        if dist > MAX_DISTANCE:
            reward -= 50
            done = True
        truncated = self.steps >= MAX_STEPS
        info = {
            'pos':tuple(self.pos.tolist()),
            'dist':float(dist),
            'steps':self.steps
        }
        return self._get_obs(), reward, done, truncated, info
    def _get_obs(self):
        return np.concatenate([self.pos, self. target]).astype(np.float32)
    def close(self):
        return super().close()


def train_ue5():
    env = DroneEnv(mode='ue5')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_SAVE_PATH)
    env.close()
    print('Hyina study')

def train():
    env = DroneEnv(mode='virtual')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_SAVE_PATH)
    env.close()
    print('Hyina study')

    
if __name__ == "__main__":
    train()