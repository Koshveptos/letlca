import socket
import json
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from config import *

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
        self.prev_dist = None
        self.smooth_pos = np.array([0.0,0.0])
        self.observation_space = spaces.Box(-1000, 1000, (4,), dtype=np.float32)
        self.action_space = spaces.Box(-MAX_SPEED, MAX_SPEED, (2,), dtype=np.float32)
    def reset(self, seed = None, options=None):
        super().reset(seed = seed)
        self.pos = np.array([0.0, 0.0])
        self.smooth_pos = np.array([0.0 , 0.0])
        self.steps = 0
        self.prev_dist = None

        if self.mode == 'ue5' and self.drone:
            self.drone.step(0,0)
        return self._get_obs(), {}
    

    def step(self, action):
        vx, vy = np.clip(action, -MAX_SPEED, MAX_SPEED)
        #virtual
        if self.mode == 'virtual':
            if UDP_DELAY > 0:
                time.sleep(UDP_DELAY * 0.5)

            noise_vx = np.random.normal(0, NOISE_STD)
            noise_vy = np.random.normal(0, NOISE_STD)
            actual_vx = vx + noise_vx
            actual_vy = vy + noise_vy


            self.pos  += np.array([actual_vx, actual_vy]) * DT
            #self.pos = np.clip(self.pos, -1000, 1000)
            self.smooth_pos = self.smooth_pos * 0.8 + self.pos * 0.2
            self.pos = self.smooth_pos
            self.pos = np.clip(self.pos, -BOUNDARY, BOUNDARY)
            if np.random.random() < 0.01:  
                pass
        # if reaaaalllll
        else:
            if self.drone:
                state = self.drone.step(vx,  vy)
                if state:
                    self.pos = np.array([state.get('X', 0), state.get('Y',0)])
        self.steps += 1



        dist = np.linalg.norm(self.pos - self.target)
        reward = -dist / 300
        ###фикс наград и наказаний 
        #если движется к цели 
        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            if delta > 1:
                reward += 2.0
            if delta < -1.0:
                reward -= 1.0
        #если хотя бы смотрит на цель

        if dist > 0:
            to_target = self.target - self.pos
            velocity = np.array([vx,vy])
            vel_mag = np.linalg.norm(velocity)
            if vel_mag > 0.1:
                cos_angle = np.dot(to_target, velocity) / (np.linalg.norm(to_target) * np.linalg.norm(vel_mag) + 0.001) 
                reward += 3 * max(0, cos_angle)


        if abs(vx) <= 1.0 and abs(vy) <= 1.0:
            reward -= 0.5

        speed = np.linalg.norm([vx,vy])
        if speed > 10:
            reward += speed / 600
        #достиг цели
        done = False
        if dist < REACH_RADIUS:
            time_bonus = max(0, (800 - self.steps) / 800) * 100
            reward += 250 + time_bonus
            done = True
            print("=" * 50)
            print(f"============ долетел  шаги - {self.steps}  расстояние - {dist}")
            print("=" * 50)


        if dist > MAX_DISTANCE:
            reward -= 150
            done = True
       
        if self.steps > 400 and dist > 300:
            reward -= 0.1
            
        self.prev_dist = dist
        
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
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        tensorboard_log=TENSORBOARD_LOG,
    )
#     eval_callback = EvalCallback(
#     env,
#     best_model_save_path='./models/',
#     log_path='./logs/',
#     eval_freq=10000,
#     deterministic=True,
#     render=False,
#     verbose=1
# )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        # callback=eval_callback,
        progress_bar=True
    )
    model.save(MODEL_SAVE_PATH)
    env.close()
    print('Hyina study')

    


def test():
    print("Тестирование агента...")
    

    try:
        model = PPO.load(MODEL_SAVE_PATH)
    except:
        print(f"Модель {MODEL_SAVE_PATH} не найдена!")
        return
    

    env = DroneEnv(mode='virtual')
    

    obs, _ = env.reset()
    total_reward = 0
    steps = 0
    

    print("-" * 60)
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        pos = info['pos']
        dist = info['dist']
        speed = info.get('speed', 0)
        

        if steps % 10 == 0:
            print(f"Шаг {steps:3d}: ({pos[0]:7.1f}, {pos[1]:7.1f}) "
                  f"расст: {dist:7.1f} скорость: {speed:5.1f} "
                  f"награда: {reward:7.2f}")
        
        if done or truncated:
            break
    
    print("-" * 60)
    print(f"\n'пизод завершён!")
    print(f"   всего шагов: {steps}")
    print(f"   Суммарная награда: {total_reward:.2f}")
    print(f"   финишная позиция: ({obs[0]:.1f}, {obs[1]:.1f})")
    print(f"   расстояние до цели: {info['dist']:.1f}")
    
    if info['dist'] < REACH_RADIUS:
        print("ЦЕЛЬ ДОСТИГНУТА!")
    
    env.close()




if __name__ == "__main__":
    train()