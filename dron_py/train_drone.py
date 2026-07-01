import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from drone_env import DroneEnv
from mission_wrapper import MissionWrapper


# ===================== CONFIG =====================
TOTAL_TIMESTEPS = 1_000_000      # Сколько шагов делать за запуск
MODEL_PATH = "drone_model.zip"
VEC_PATH = "vecnorm.pkl"
LOG_DIR = "./logs/"


def make_env():
    return MissionWrapper(DroneEnv(chain_prob=0.62))


# ===================== MAIN =====================
if __name__ == "__main__":
    print("=== Инициализация окружения ===")
    
    env = DummyVecEnv([make_env])
    
    # Загрузка нормализации
    if os.path.exists(VEC_PATH):
        print(f"Загружаем нормализацию из {VEC_PATH}")
        env = VecNormalize.load(VEC_PATH, env)
        env.training = True
        env.norm_reward = True
    else:
        print("Создаём новую нормализацию")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Загрузка модели или создание новой
    if os.path.exists(MODEL_PATH):
        print(f"Загружаем существующую модель: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
        print("→ Продолжаем обучение...")
    else:
        print("Создаём новую модель PPO")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2.7e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.993,
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.0065,        # хороший баланс
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=LOG_DIR
        )

    print(f"=== Запуск обучения на {TOTAL_TIMESTEPS:,} шагов ===")
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=False     # важно для продолжения
    )

    # Сохранение
    model.save(MODEL_PATH)
    env.save(VEC_PATH)
    print(f"=== Обучение завершено! Модель сохранена в {MODEL_PATH} ===")