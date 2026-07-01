import gymnasium as gym
import numpy as np


class MissionWrapper(gym.Wrapper):
    def __init__(self, env, p_chain=0.65):
        super().__init__(env)
        self.p_chain = p_chain

    def reset(self, seed=None, options=None):
        if options is not None and "waypoints" in options:
            return self.env.reset(seed=seed, options=options)

        if np.random.rand() < self.p_chain:
            n = np.random.randint(3, 8)
            waypoints = np.random.uniform(0, 1000, (n, 2)).astype(np.float32)
            start = np.random.uniform(0, 1000, 2).astype(np.float32)

            return self.env.reset(seed=seed, options={
                "start": start,
                "waypoints": waypoints
            })
        else:
            return self.env.reset(seed=seed)

    def step(self, action):

        action = np.array(action, dtype=np.float32).flatten()
        
        # Если action пришёл только с одним значением (баг VecEnv)
        if len(action) == 1:
            # Это ошибка, но на всякий случай
            action = np.array([action[0], 0.0], dtype=np.float32)
        
        return self.env.step(action)