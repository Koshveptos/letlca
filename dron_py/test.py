import numpy as np
import os
from drone_client import DroneEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config import *

def test_scenario(start_pos, target_pos, title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"   Start: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
    print(f"   Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
    print('='*60)
    
    # 1. Создаем среду
    env = DroneEnv(mode='virtual')
    env = DummyVecEnv([lambda: env])
    
    # 2. ✅ ЗАГРУЖАЕМ НОРМАЛИЗАЦИЮ (ОБЯЗАТЕЛЬНО!)
    if os.path.exists("vec_normalize.pkl"):
        env = VecNormalize.load("vec_normalize.pkl", env)
        env.training = False      # Отключаем обновление статистик
        env.norm_reward = False   # Не нормализуем награду при тесте
    else:
        print("⚠️ ВНИМАНИЕ: vec_normalize.pkl не найден! Модель будет работать плохо.")
    
    # 3. Загружаем модель
    model = PPO.load(MODEL_SAVE_PATH)
    
    # 4. Сбрасываем среду с нужными координатами
    obs = env.reset(
    )
    
    total_reward = 0
    steps = 0
    done = [False]
    truncated = [False]
    
    while not done[0] and not truncated[0] and steps < 800:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # В VecEnv reward и done - это массивы, берём первый элемент
        total_reward += reward[0]
        steps += 1
        
        if steps % 20 == 0:
            # Получаем реальные (денормализованные) наблюдения
            real_obs = env.get_original_obs()[0]
            real_info = info[0]
            print(f"Step {steps:3d}: position ({real_obs[0]*1000:7.1f}, {real_obs[1]*1000:7.1f}) "
                  f"dist: {real_info['dist']:7.1f} yaw: {real_obs[4]*360:6.1f} deg")
    
    # Проверяем успех
    final_info = info[0]
    success = final_info['dist'] < REACH_RADIUS
    
    real_obs = env.get_original_obs()[0]
    print(f"\nResult:")
    print(f"   Steps: {steps}")
    print(f"   Reward: {total_reward:.2f}")
    print(f"   Finish: ({real_obs[0]*1000:.1f}, {real_obs[1]*1000:.1f})")
    print(f"   Distance to target: {final_info['dist']:.1f}")
    print(f"   Yaw: {real_obs[4]*360:.1f} deg")
    print(f"   {'✅ SUCCESS' if success else '❌ FAIL'}")
    
    env.close()
    return success

if __name__ == "__main__":
    scenarios = [
        ((0, 0), (500, 300), "Test 1: (0,0) -> (500,300)"),
        ((200, 100), (500, 300), "Test 2: (200,100) -> (500,300)"),
        ((300, 200), (500, 300), "Test 3: (300,200) -> (500,300)"),
        ((0, 0), (400, 200), "Test 4: (0,0) -> (400,200)"),
        ((100, 100), (100, 100), "Test 5: (100,100) -> (100,100)"),
        ((300, 200), (100, 50), "Test 6: (300,200) -> (100,50)"),
    ]
    
    print("TESTING AGENT WITH DIFFERENT SCENARIOS")
    print("Model:", MODEL_SAVE_PATH)
    
    results = []
    for start, target, title in scenarios:
        success = test_scenario(start, target, title)
        results.append(success)
    
    print("\n" + "="*60)
    print(f"TOTAL: {sum(results)}/{len(results)} successful")
    print("="*60)