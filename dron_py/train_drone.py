import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from drone_env import DroneEnv   
from config import *


def make_env():
    return DroneEnv()



import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from drone_env import DroneEnv   # твоя среда
from config import *


def make_env():
    return DroneEnv()


if __name__ == "__main__":


    env = DummyVecEnv([make_env])

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0
    )

 

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        training=False
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path="logs/",
        eval_freq=10_000,
        deterministic=True
    )

    #model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        tensorboard_log=LOG_DIR
    )


    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback
    )


    model.save(MODEL_PATH)

    env.save(VEC_PATH)

    print("Training finished")
