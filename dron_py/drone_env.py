import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import *
from utils import random_scenario, clamp_position


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # [forward_thrust, yaw_rate]
        self.action_space = spaces.Box( # определение действий агента 
            low=np.array([0.0, -1.0], dtype=np.float32), # сила тяги
            high=np.array([1.0, 1.0], dtype=np.float32), # скорость поворота  left right 
            dtype=np.float32,
        )

        
        self.observation_space = spaces.Box( # определение что видит агент
            low=-1.0,
            high=1.0,
            shape=(9,), # 9 числе из obs, каждое от -1 до 1
            dtype=np.float32,
        )

        # stat
        # zeros - нулевка матрица 
        self.position = np.zeros(2, dtype=np.float32)  # 2 значения глобал коорд
        self.velocity = np.zeros(2, dtype=np.float32) # 2 значения вектор скорости по x y 

        self.target = np.zeros(2, dtype=np.float32) # аналогично координаты цели 

        self.yaw = 0.0  

        self.prev_distance = 0.0 # пред расстояние для расчета прогресса 
        self.current_step = 0
        self.success = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.success = False
        self.velocity[:] = 0

        self.position, self.target = random_scenario()

        self.yaw = np.random.uniform(-180, 180)

        self.prev_distance = np.linalg.norm(self.target - self.position)

        return self._get_observation(), {
            "position_global": self.position.copy(),
            "target_global": self.target.copy(),
            "yaw": self.yaw,
        }


    def step(self, action):

        self.current_step += 1

        thrust = np.clip(action[0], 0.0, 1.0)
        yaw_rate = np.clip(action[1], -1.0, 1.0)

        # new rot
        self.yaw += yaw_rate * 5.0
        self.yaw %= (self.yaw + 180) % 360 - 180

        #vector to targ
        yaw_rad = np.deg2rad(self.yaw)

        forward = np.array([
            np.cos(yaw_rad),
            np.sin(yaw_rad)
        ], dtype=np.float32)

        # ph
        self.velocity = forward * thrust * MAX_SPEED

        self.position += self.velocity * DT
        self.position = clamp_position(self.position)

      
        to_target = self.target - self.position
        dist = np.linalg.norm(to_target)

        #награды
        reward = 0.0

       
        reward -= dist / 1000.0

        
        progress = self.prev_distance - dist
        reward += progress * 2.0

        
        norm_t = np.linalg.norm(to_target) + 1e-8
        norm_v = np.linalg.norm(self.velocity) + 1e-8

        alignment = np.dot(to_target, self.velocity) / (norm_t * norm_v)
        reward += 0.5 * alignment

    
        local_target = self._world_to_local(to_target)
        angle_to_target = np.arctan2(local_target[1], local_target[0])
        reward += 0.2 * np.cos(angle_to_target)

      
        terminated = False
        if dist < REACH_RADIUS:
            reward += 100.0
            terminated = True
            self.success = True

      
        truncated = self.current_step >= MAX_STEPS

        self.prev_distance = dist

        return self._get_observation(), reward, terminated, truncated, self._get_info(dist)

    
    def _get_observation(self):

        to_target = self.target - self.position
        local = self._world_to_local(to_target)

        obs = np.array([
            self.position[0] / BOUNDARY,
            self.position[1] / BOUNDARY,

            self.target[0] / BOUNDARY,
            self.target[1] / BOUNDARY,

            local[0] / BOUNDARY,
            local[1] / BOUNDARY,

            np.cos(np.deg2rad(self.yaw)),
            np.sin(np.deg2rad(self.yaw)),

            np.linalg.norm(self.velocity) / MAX_SPEED,
        ], dtype=np.float32)

        return obs

    
    def _world_to_local(self, vec):

        yaw_rad = np.deg2rad(self.yaw)

        c = np.cos(-yaw_rad)
        s = np.sin(-yaw_rad)

        rot = np.array([
            [c, -s],
            [s,  c]
        ], dtype=np.float32)

        return rot @ vec


    def _get_info(self, dist):

        local_target = self._world_to_local(self.target - self.position)

        return {
            
            "position_global": self.position.copy(),
            "target_global": self.target.copy(),

            
            "target_local": local_target,

           
            "distance": float(dist),
            "yaw": float(self.yaw),
            "steps": self.current_step,
        }

   
    def close(self):
        return super().close()