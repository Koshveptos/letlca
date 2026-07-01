import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env import DroneEnv
from mission_wrapper import MissionWrapper
from config import BOUNDARY, MAX_STEPS


MODEL_PATH = "drone_model.zip"
VEC_PATH = "vecnorm.pkl"


def make_env():
    env = DroneEnv()
    env = MissionWrapper(env, p_chain=0.0)  # только single
    return env


def load_env():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_PATH, env)
    env.training = False
    env.norm_reward = False
    return env


def test_single():
    print("\n======================")
    print("SINGLE TEST")
    print("======================")

    env = load_env()
    model = PPO.load(MODEL_PATH)

    obs = env.reset()

    e = env.envs[0].env.env  # unwrap: Vec -> Wrapper -> Env

    target = e.target.copy()

    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            break

    print("TARGET GLOBAL:", target)
    print("FINAL POS    :", e.position)
    print("DIST         :", np.linalg.norm(e.position - target))
    print("STEPS        :", step)


if __name__ == "__main__":
    test_single()