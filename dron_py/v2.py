
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
        
    def send(self, vx, vy, yaw_rate=0.0):
         msg = json.dumps({"Vx": float(vx), "Vy": float(vy), 'YawRate':float(yaw_rate)})
         self.sock.sendto(msg.encode('utf-8'), (self.ip, self.send_port))

    def receive(self):
        if not self.sock:
            return None
        try:
            data, _ = self.sock.recvfrom(1024)
            return json.loads(data.decode('utf-8'))
        except :
            return  None
        
    def one_step(self, vx, vy, yaw_rate = 0.0):
        self.send(vx,vy, yaw_rate)
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

        self.target = np.array([TARGET_X, TARGET_Y], dtype=np.float64)
        self.pos = np.array([0.0, 0.0], dtype=np.float64)
        self.steps = 0
        self.prev_dist = None
        self.yaw = INITIAL_YAW
        self.smooth_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float64)

        self.observation_space = spaces.Box(-1000, 1000, shape=(7,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-MAX_SPEED, -MAX_SPEED, -MAX_YAW_RATE], dtype=np.float32),
            high=np.array([MAX_SPEED, MAX_SPEED, MAX_YAW_RATE], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Если переданы options, используем их
        if options and 'start_pos' in options and 'target_pos' in options:
            self.pos = np.array(options['start_pos'], dtype=np.float64)
            self.target = np.array(options['target_pos'], dtype=np.float64)
        else:
            # Случайная инициализация
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(50, 500)
            self.pos = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64)
            
            target_angle = np.random.uniform(0, 2 * np.pi)
            target_radius = np.random.uniform(50, 500)
            self.target = np.array([target_radius * np.cos(target_angle), target_radius * np.sin(target_angle)], dtype=np.float64)
        
      
        self.smooth_pos = self.pos.copy()
        self.yaw = INITIAL_YAW
        self.steps = 0
        self.prev_dist = None
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float64)
        
        if self.mode == 'ue5' and self.drone:
            self.drone.one_step(0, 0, 0)
        
        return self._get_obs(), {}

    def set_scenario(self, start_pos, target_pos):
        """Явный метод для установки сценария"""
        self.pos = np.array(start_pos, dtype=np.float64)
        self.target = np.array(target_pos, dtype=np.float64)
        self.smooth_pos = self.pos.copy()  
        self.yaw = INITIAL_YAW
        self.steps = 0
        self.prev_dist = None
        self.prev_velocity = np.array([0.0, 0.0], dtype=np.float64)

    def step(self, action):
        vx, vy, yaw_rate = np.clip(
            action,
            [-MAX_SPEED, -MAX_SPEED, -MAX_YAW_RATE],
            [MAX_SPEED, MAX_SPEED, MAX_YAW_RATE]
        )

        self.yaw += yaw_rate * DT
        self.yaw = self.yaw % 360

        if self.mode == 'virtual':
            if UDP_DELAY > 0:
                time.sleep(UDP_DELAY * 0.5)

            noise_vx = np.random.normal(0, NOISE_STD)
            noise_vy = np.random.normal(0, NOISE_STD)
            actual_vx = vx + noise_vx
            actual_vy = vy + noise_vy

            self.pos = self.pos + np.array([actual_vx, actual_vy], dtype=np.float64) * DT
            self.smooth_pos = self.smooth_pos * 0.8 + self.pos * 0.2
            self.pos = self.smooth_pos
            self.pos = np.clip(self.pos, -BOUNDARY, BOUNDARY)
        else:
            if self.drone:
                state = self.drone.one_step(vx, vy, yaw_rate)
                if state:
                    self.pos = np.array([state.get('X', 0.0), state.get('Y', 0.0)], dtype=np.float64)
                    if 'Yaw' in state:
                        self.yaw = state.get('Yaw', 0) % 360

        self.steps += 1

       
        dist = np.linalg.norm(self.pos - self.target)
        
      
        reward = -dist * 0.01
        
   
        if self.prev_dist is not None:
            delta = self.prev_dist - dist
            reward += delta * 0.5  
        
    
        reward -= 0.1

        reward -= 0.0001 * (vx**2 + vy**2 + yaw_rate**2)

        done = False
        

        if dist < REACH_RADIUS:
            reward += 100.0 + max(0, (MAX_STEPS - self.steps) / MAX_STEPS) * 50
            done = True
            print("=" * 50)
            print(f" Долетел! Шаг {self.steps}, расстояние {dist:.1f}")
            print("=" * 50)

     
        if dist > MAX_DISTANCE:
            reward -= 100.0
            done = True

        self.prev_dist = dist
        truncated = self.steps >= MAX_STEPS

        info = {
            'pos': tuple(self.pos.tolist()),
            'dist': float(dist),
            'steps': self.steps,
            'speed': float(np.linalg.norm([vx, vy]))
        }

        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        return np.array([
            self.pos[0] / 1000.0,
            self.pos[1] / 1000.0,
            self.target[0] / 1000.0,
            self.target[1] / 1000.0,
            self.yaw / 360,
            self.prev_velocity[0] / MAX_SPEED if hasattr(self, 'prev_velocity') else 0.0,
            self.prev_velocity[1] / MAX_SPEED if hasattr(self, 'prev_velocity') else 0.0
        ], dtype=np.float32)

    def close(self):
        if self.mode == 'ue5' and self.drone:
            self.drone.close_connection()
        return super().close()
    

def train_ue5():
    env = DroneEnv(mode='ue5')
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(MODEL_SAVE_PATH)
    env.close()
    print('Hyina study')


def train():
    def make_env():
        def _init():
            env = DroneEnv(mode='virtual')
            return env
        return _init
    
  
    env = DummyVecEnv([make_env()])
    

    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1000.)
    
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=TENSORBOARD_LOG,
    )
    
    
    print("Начало обучения (2 миллиона шагов)...")
    model.learn(total_timesteps=400_000, progress_bar=True)
    
    
    model.save(MODEL_SAVE_PATH)
    env.save("vec_normalize.pkl")
    env.close()
    print(' Обучение завершено!')
    
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