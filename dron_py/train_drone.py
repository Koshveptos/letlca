import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from drone_env import DroneEnv
from mission_wrapper import MissionWrapper
from config import TOTAL_TIMESTEPS, MODEL_PATH, VEC_PATH


# ---------------- ENV FACTORY ----------------
def make_env():
    env = DroneEnv()
    env = MissionWrapper(env, p_chain=0.5)  # 50% chain / 50% single
    return env


# ---------------- MAIN ----------------
def main():

    env = DummyVecEnv([make_env])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

    # checkpoint saving (очень важно, иначе ты потеряешь обучение)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints/",
        name_prefix="drone"
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,

        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,

        tensorboard_log="logs/"
    )

    print("\n==============================")
    print("START TRAINING DRONE PPO")
    print("==============================\n")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )

    # ---------------- SAVE ----------------
    os.makedirs("models", exist_ok=True)

    model.save(MODEL_PATH)
    env.save(VEC_PATH)

    print("\n==============================")
    print("TRAINING FINISHED")
    print("MODEL SAVED:", MODEL_PATH)
    print("VEC SAVED  :", VEC_PATH)
    print("==============================\n")


if __name__ == "__main__":
    main()