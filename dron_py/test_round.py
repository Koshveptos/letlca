import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drone_env import DroneEnv
from mission_wrapper import MissionWrapper
from config import MAX_STEPS, BOUNDARY, REACH_RADIUS


MODEL_PATH = "drone_model.zip"
VEC_PATH = "vecnorm.pkl"


def make_env():
    return MissionWrapper(DroneEnv())


def load_env():
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_PATH, env)
    env.training = False
    env.norm_reward = False
    return env

def run_test(env, model, chain=False, episodes=1):
    for ep in range(episodes):
        print(f"\n=== {'CHAIN' if chain else 'SINGLE'} TEST {ep+1} ===")
        
        base = env.envs[0].env  # оригинальный DroneEnv
        
        if chain:
            n = np.random.randint(4, 7)
            waypoints = np.random.uniform(80, BOUNDARY-80, (n, 2)).astype(np.float32)
            obs, _ = base.reset(options={
                "start": np.random.uniform(50, BOUNDARY-50, 2),
                "waypoints": waypoints
            })
            # Пересинхронизация нормализации
            obs = env.reset()[0]
        else:
            obs = env.reset()[0]

        total_reward = 0.0
        done = False
        step = 0

        while step < MAX_STEPS and not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            step += 1

        # Итог
        final_pos = base.position.copy()
        final_target = base.final_target.copy()
        last_target = base.target.copy()
        dist_final = np.linalg.norm(final_pos - final_target)
        dist_last = np.linalg.norm(final_pos - last_target)

        print(f"Steps          : {step}")
        print(f"Final dist     : {dist_final:.2f}")
        print(f"Last WP dist   : {dist_last:.2f}")
        print(f"Total reward   : {total_reward:.2f}")
        print(f"Success (final): {dist_final < REACH_RADIUS + 30}")
        print("-" * 50)

        
def test():
    model = PPO.load(MODEL_PATH)
    env = load_env()

    print("=== SINGLE TARGET TEST ===")
    run_test(env, model, chain=False)

    print("\n=== CHAIN MISSION TEST ===")
    run_test(env, model, chain=True, episodes=1)


if __name__ == "__main__":
    test()