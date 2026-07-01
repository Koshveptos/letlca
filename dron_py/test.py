import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env import DroneEnv
from config import *


def make_env():
    return DroneEnv()


def run_episode(env, model, episode_id):
    obs = env.reset()

    start = info = env.get_attr("position")[0] if hasattr(env, "get_attr") else None


    raw_env = env.envs[0]

    start_pos = raw_env.position.copy()
    target_pos = raw_env.target.copy()

    print("\n" + "-" * 60)
    print(f"EPISODE {episode_id}")
    print(f"START  : {start_pos}")
    print(f"TARGET : {target_pos}")
    print(f"DIST   : {np.linalg.norm(target_pos - start_pos):.2f}")
    print("-" * 60)

    total_reward = 0

    for step in range(MAX_STEPS):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        total_reward += reward[0]

        if step % 20 == 0:
            print(
                f"step {step:3d} | "
                f"pos {info[0]['position']} | "
                f"dist {info[0]['distance']:.2f}"
            )

        if done[0]:
            break

    final = info[0]

    print("\nRESULT:")
    print(f"END POS : {final['position']}")
    print(f"TARGET  : {target_pos}")
    print(f"FINAL DIST: {final['distance']:.2f}")
    print(f"SUCCESS : {final['distance'] < REACH_RADIUS}")
    print(f"REWARD  : {total_reward:.2f}")

    return final['distance'] < REACH_RADIUS



if __name__ == "__main__":

    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_PATH, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(MODEL_PATH)

    print("\n" + "=" * 60)
    print("RANDOM SCENARIO TEST WITH START/GOAL TRACE")
    print("=" * 60)

    results = []

    for i in range(10):
        success = run_episode(env, model, i + 1)
        results.append(success)

    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    print(f"Success rate: {sum(results)}/10")
    print("=" * 60)