import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import BOUNDARY, MAX_SPEED, DT, MAX_STEPS, REACH_RADIUS


class DroneEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0], dtype=np.float32),
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )

        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

        self.yaw = 0.0
        self.prev_distance = 0.0
        self.last_thrust = 0.0
        self.step_count = 0

        # mission
        self.waypoints = None
        self.wp_idx = 0
        self.mission_mode = False

    # ---------------- RESET ----------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.velocity[:] = 0
        self.last_thrust = 0.0
        self.yaw = np.random.uniform(-180, 180)

        # mission mode
        if options and "waypoints" in options:
            self.mission_mode = True
            self.waypoints = np.array(options["waypoints"], dtype=np.float32)
            self.wp_idx = 0

            self.position = np.array(options.get(
                "start",
                np.random.uniform(0, BOUNDARY, 2)
            ), dtype=np.float32)

            self.target = self.waypoints[0]

        else:
            self.mission_mode = False
            self.waypoints = None
            self.wp_idx = 0

            self.position = np.random.uniform(0, BOUNDARY, 2)
            self.target = np.random.uniform(0, BOUNDARY, 2)

        self._clip()
        self.prev_distance = self._dist()

        return self._obs(), self._info()

    # ---------------- STEP ----------------
    def step(self, action):
        self.step_count += 1

        thrust = float(np.clip(action[0], 0, 1))
        yaw_rate = float(np.clip(action[1], -1, 1))

        self.last_thrust = thrust

        self.yaw += yaw_rate * 5.0
        self.yaw = (self.yaw + 180) % 360 - 180

        yaw = np.deg2rad(self.yaw)

        direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float32)

        self.velocity = direction * thrust * MAX_SPEED
        self.position += self.velocity * DT

        self._clip()

        dist = self._dist()
        to_target = self.target - self.position

        reward = 0.0
        reward += (self.prev_distance - dist) * 6.0
        reward -= dist / BOUNDARY

        vel = np.linalg.norm(self.velocity)
        if vel > 1e-6:
            reward += 0.3 * np.dot(
                self.velocity / vel,
                to_target / (dist + 1e-8)
            )

        terminated = False
        truncated = self.step_count >= MAX_STEPS

        # ---------------- WAYPOINT LOGIC (INSIDE ENV) ----------------
        if self.mission_mode and dist < REACH_RADIUS:
            reward += 60.0

            self.wp_idx += 1

            if self.wp_idx >= len(self.waypoints):
                terminated = True
            else:
                self.target = self.waypoints[self.wp_idx]
                self.prev_distance = self._dist()
                self.velocity[:] = 0

        elif dist < REACH_RADIUS:
            reward += 60.0
            terminated = True

        self.prev_distance = dist

        return self._obs(), reward, terminated, truncated, self._info()

    # ---------------- OBS ----------------
    def _obs(self):
        to_target = self.target - self.position
        local = self._world_to_local(to_target)

        return np.array([
            local[0] / BOUNDARY,
            local[1] / BOUNDARY,
            np.cos(np.deg2rad(self.yaw)),
            np.sin(np.deg2rad(self.yaw)),
            self.last_thrust
        ], dtype=np.float32)

    def _world_to_local(self, v):
        yaw = np.deg2rad(self.yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        return np.array([[c, -s], [s, c]]) @ v

    def _dist(self):
        return np.linalg.norm(self.target - self.position)

    def _clip(self):
        self.position = np.clip(self.position, 0, BOUNDARY)

    def _info(self):
        return {
            "pos": self.position.copy(),
            "target": self.target.copy(),
            "dist": float(self._dist()),
            "yaw": float(self.yaw),
            "wp": self.wp_idx
        }