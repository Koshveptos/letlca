import gymnasium as gym
import numpy as np


class MissionWrapper(gym.Wrapper):

    def __init__(self, env, p_chain=0.5):
        super().__init__(env)
        self.p_chain = p_chain

    def reset(self, seed=None, options=None):
        if np.random.rand() < self.p_chain:

            n = np.random.randint(3, 10)

            waypoints = np.random.uniform(0, 1000, (n, 2)).astype(np.float32)

            obs, info = self.env.reset(options={
                "start": np.random.uniform(0, 1000, 2),
                "waypoints": waypoints
            })

        else:
            obs, info = self.env.reset()

        return obs, info

    def step(self, action):
        return self.env.step(action)