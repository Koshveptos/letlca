import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import BOUNDARY, MAX_SPEED, DT, MAX_STEPS, REACH_RADIUS


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, chain_prob=0.6):
        super().__init__()
        self.chain_prob = chain_prob

        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        # State
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0

        self.target = np.zeros(2, dtype=np.float32)
        self.final_target = np.zeros(2, dtype=np.float32)

        self.waypoints = None
        self.wp_idx = 0
        self.mission_mode = False

        self.step_count = 0
        self.prev_distance = 0.0
        self.prev_final_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.velocity[:] = 0.0
        self.yaw = np.random.uniform(-180, 180)

        self.waypoints = None
        self.wp_idx = 0
        self.mission_mode = False

        if options and "waypoints" in options:
            self.mission_mode = True
            self.waypoints = np.array(options["waypoints"], dtype=np.float32)
            self.position = np.array(options.get("start", np.random.uniform(0, BOUNDARY, 2)), dtype=np.float32)
            self.target = self.waypoints[0]
            self.final_target = self.waypoints[-1]
        else:
            if np.random.rand() < self.chain_prob:
                self.mission_mode = True
                n = np.random.randint(3, 8)
                self.waypoints = np.random.uniform(0, BOUNDARY, (n, 2)).astype(np.float32)
                self.position = np.random.uniform(0, BOUNDARY, 2).astype(np.float32)
                self.target = self.waypoints[0]
                self.final_target = self.waypoints[-1]
            else:
                self.position = np.random.uniform(0, BOUNDARY, 2).astype(np.float32)
                self.target = np.random.uniform(0, BOUNDARY, 2).astype(np.float32)
                self.final_target = self.target

        self.prev_distance = self._dist()
        self.prev_final_distance = np.linalg.norm(self.final_target - self.position)
        
        return self._obs(), self._info()

    def step(self, action):
        self.step_count += 1
        action = np.array(action, dtype=np.float32).flatten()
        if len(action) == 1:
            action = np.array([action[0], 0.0], dtype=np.float32)

        thrust = float(np.clip(action[0], 0.0, 1.0))
        yaw_rate = float(np.clip(action[1], -1.0, 1.0))

        # Yaw
        self.yaw += yaw_rate * 6.0
        self.yaw = (self.yaw + 180) % 360 - 180
        yaw_rad = np.deg2rad(self.yaw)

        direction = np.array([np.cos(yaw_rad), np.sin(yaw_rad)], dtype=np.float32)

        # Physics
        acceleration = direction * thrust * 160.0
        self.velocity += acceleration * DT
        self.velocity *= 0.93
        speed = np.linalg.norm(self.velocity)
        if speed > MAX_SPEED:
            self.velocity *= MAX_SPEED / speed

        self.position += self.velocity * DT
        self.position = np.clip(self.position, 0, BOUNDARY)

        dist_current = self._dist()
        dist_final = np.linalg.norm(self.final_target - self.position)
        progress = self.prev_distance - dist_current
        final_progress = self.prev_final_distance - dist_final

        # ==================== REWARD ====================
        reward = progress * 5.5                    # прогресс к текущей цели
        reward += final_progress * 2.0             # прогресс к финальной цели

        # Alignment
        if speed > 15.0:
            vel_dir = self.velocity / speed
            to_target = self.target - self.position
            tgt_norm = np.linalg.norm(to_target)
            if tgt_norm > 25.0:
                reward += 0.9 * np.dot(vel_dir, to_target / tgt_norm)

        reward += thrust * 1.1
        reward -= dist_current / 1100.0
        reward -= dist_final / 2000.0
        reward -= 0.002 * self.step_count

        terminated = False
        truncated = self.step_count >= MAX_STEPS

        if dist_current < REACH_RADIUS:
            reward += 220.0
            if self.mission_mode:
                if self._next_wp():
                    reward += 420.0
                    terminated = True
                else:
                    reward += 110.0
            else:
                terminated = True

        self.prev_distance = dist_current
        self.prev_final_distance = dist_final

        return self._obs(), reward, terminated, truncated, self._info()

    def _next_wp(self):
        self.wp_idx += 1
        if self.wp_idx >= len(self.waypoints):
            return True
        self.target = self.waypoints[self.wp_idx]
        self.prev_distance = self._dist()
        self.velocity *= 0.35   # мягкое торможение
        return False

    def _obs(self):
        to_target = self.target - self.position
        local_target = self._world_to_local(to_target)
        local_vel = self._world_to_local(self.velocity)
        
        dist_current = self._dist()
        dist_final = np.linalg.norm(self.final_target - self.position)
        
        wp_progress = 0.0
        if self.mission_mode and len(self.waypoints) > 1:
            wp_progress = self.wp_idx / (len(self.waypoints) - 1)

        return np.array([
            local_target[0] / BOUNDARY,
            local_target[1] / BOUNDARY,
            local_vel[0] / MAX_SPEED,
            local_vel[1] / MAX_SPEED,
            np.cos(np.deg2rad(self.yaw)),
            np.sin(np.deg2rad(self.yaw)),
            dist_current / BOUNDARY,
            dist_final / BOUNDARY,      # важно для финальной цели
            wp_progress
        ], dtype=np.float32)

    def _world_to_local(self, vec):
        yaw = np.deg2rad(self.yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        return np.array([[c, -s], [s, c]], dtype=np.float32) @ vec

    def _dist(self):
        return np.linalg.norm(self.target - self.position)

    def _info(self):
        return {
            "pos": self.position.copy(),
            "target": self.target.copy(),
            "final": self.final_target.copy(),
            "dist": float(self._dist()),
            "dist_final": float(np.linalg.norm(self.final_target - self.position)),
            "wp": self.wp_idx,
            "chain": self.mission_mode
        }