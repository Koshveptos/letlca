import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env import DroneEnv
from config import MAX_STEPS, VEC_PATH, MODEL_PATH


def make_env():
    return DroneEnv()


def build_env():
    env = DummyVecEnv([make_env])

    env = VecNormalize.load(VEC_PATH, env)
    env.training = False
    env.norm_reward = False

    return env


def test_waypoints():
    print("=" * 60)
    print("MULTI-WAYPOINT TEST")
    print("=" * 60)

    env = build_env()
    model = PPO.load(MODEL_PATH)

    model.set_env(env)

    num_waypoints = 12

    waypoints = [
        np.array([np.random.uniform(50, 950),
                  np.random.uniform(50, 950)], dtype=np.float32)
        for _ in range(num_waypoints)
    ]

    start = np.array([832.89, 605.99], dtype=np.float32)

    obs = env.reset()

    # unwrap VecEnv
    env.envs[0].position = start.copy()
    env.envs[0].target = waypoints[0].copy()
    env.envs[0].yaw = np.random.uniform(-180, 180)
    env.envs[0].velocity[:] = 0
    env.envs[0].current_step = 0
    env.envs[0].prev_distance = np.linalg.norm(
        env.envs[0].target - env.envs[0].position
    )

    total_steps = 0

    for i, wp in enumerate(waypoints):

        env.envs[0].target = wp.copy()
        env.envs[0].current_step = 0

        done = False
        steps = 0

        while not done and steps < MAX_STEPS:

            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            steps += 1
            total_steps += 1

            done = done[0]

        pos = env.envs[0].position
        print("-" * 40)
        print(f"WAYPOINT {i + 1}")
        print(f"POS    : {pos}")
        print(f"TARGET : {wp}")
        print(f"STEPS  : {steps}")

    env.close()

    print("=" * 60)
    print(f"TOTAL STEPS: {total_steps}")
    print("=" * 60)


if __name__ == "__main__":
    test_waypoints()