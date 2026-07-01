import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env import DroneEnv
from mission_wrapper import MissionWrapper
from config import MODEL_PATH, VEC_PATH, MAX_STEPS, BOUNDARY


# ---------------- LOAD ----------------
def make_env():
    return MissionWrapper(DroneEnv())


def load_env():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_PATH, env)
    env.training = False
    env.norm_reward = False
    return env


# ---------------- SAFE ACTION ----------------
def fix_action(action):
    return np.array(action, dtype=np.float32).reshape(1, 2)


# ---------------- UNWRAP ----------------
def get_base_env(env):
    base = env.envs[0]
    if hasattr(base, "env"):
        return base.env
    return base


# ---------------- SINGLE TEST ----------------
def test_single():
    print("\n" + "=" * 60)
    print("SINGLE TEST")
    print("=" * 60)

    env = load_env()
    model = PPO.load(MODEL_PATH)

    obs = env.reset()
    e = get_base_env(env)

    target = e.target.copy()

    total_reward = 0

    for step in range(MAX_STEPS):

        action, _ = model.predict(obs, deterministic=True)
        action = fix_action(action)

        obs, reward, done, info = env.step(action)

        total_reward += reward[0]

        if done[0]:
            break

    final = e.position.copy()
    dist = np.linalg.norm(final - target)

    print("\nRESULT:")
    print("TARGET       :", target)
    print("FINAL POS    :", final)
    print("DIST         :", round(dist, 2))
    print("REACHED      :", dist < 30)
    print("SUCCESS      :", "YES" if dist < 30 else "NO")
    print("TOTAL REWARD :", round(total_reward, 2))
    print("STEPS        :", step)


# ---------------- CHAIN TEST ----------------
def test_chain():
    print("\n" + "=" * 60)
    print("CHAIN TEST")
    print("=" * 60)

    env = load_env()
    model = PPO.load(MODEL_PATH)

    waypoints = [
        np.random.uniform(0, BOUNDARY, 2).astype(np.float32)
        for _ in range(10)
    ]

    obs = env.reset()
    e = get_base_env(env)

    e.position = np.random.uniform(0, BOUNDARY, 2)
    e.waypoints = np.array(waypoints)
    e.wp_idx = 0
    e.target = waypoints[0]
    e.mission_mode = True

    total_steps = 0

    for i, wp in enumerate(waypoints):

        print("\nWAYPOINT", i + 1)
        print("TARGET:", wp)

        reached = False

        for step in range(MAX_STEPS):

            action, _ = model.predict(obs, deterministic=True)
            action = fix_action(action)

            obs, reward, done, info = env.step(action)

            total_steps += 1

            dist = np.linalg.norm(e.position - wp)

            if dist < 30:
                reached = True
                break

            if done[0]:
                break

        print("FINAL POS :", e.position)
        print("DIST      :", round(dist, 2))
        print("REACHED   :", reached)
        print("WP IDX    :", e.wp_idx)

    print("\nTOTAL STEPS:", total_steps)
    print("=" * 60)


# ---------------- RUN ----------------
if __name__ == "__main__":
    test_single()
    test_chain()