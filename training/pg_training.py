# pg_training.py
import gymnasium as gym
from stable_baselines3 import PPO
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from environment.language_translation_env import LanguageTranslationEnv

# Initialize environment
env = LanguageTranslationEnv()

# Define the PPO model with improved hyperparameters
model = PPO('MlpPolicy', env, verbose=1, learning_rate=1e-4, gamma=0.99, 
            batch_size=64, n_steps=2048, ent_coef=0.01, gae_lambda=0.95, 
            clip_range=0.2, policy_kwargs={'net_arch': [64, 64]})

# Train the model
model.learn(total_timesteps=50000)  # You can increase this for better performance

# Save the trained model
model_path = os.path.join("models", "ppo_language_translation.zip")
model.save(model_path)

print("PPO Model Training Complete!")
