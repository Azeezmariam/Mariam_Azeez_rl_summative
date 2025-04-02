import gymnasium as gym
from stable_baselines3 import DQN
import os
import sys

# Ensure Python finds the environment module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.language_translation_env import LanguageTranslationEnv

# Initialize environment
env = LanguageTranslationEnv()

# Define the DQN model with optimized hyperparameters
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=5e-4,  # Increased for faster convergence
    gamma=0.99,  # Discount factor
    batch_size=128,  # Increased batch size for better training
    buffer_size=200000,  # Larger buffer to store more experiences
    learning_starts=10000,  # Let the agent collect experience before learning
    target_update_interval=1000,  # More frequent updates to stabilize learning
    train_freq=4,  # Train every 4 steps
    gradient_steps=2,  # Perform 2 gradient steps per training
    exploration_fraction=0.15,  # Shorter exploration period
    exploration_initial_eps=1.0,  # Start with full exploration
    exploration_final_eps=0.01,  # Reduce to minimal exploration
    policy_kwargs={"net_arch": [128, 128]},  # Deeper network for better learning
)

# Train the model
model.learn(total_timesteps=100000)  # Increased training steps

# Save the trained model
model_path = os.path.join("models", "dqn_language_translation.zip")
model.save(model_path)

print("DQN Model Training Complete!")
