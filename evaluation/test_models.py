# evaluation/test_models.py
import os
from stable_baselines3 import PPO, DQN 
from environment.language_translation_env import LanguageTranslationEnv

# Initialize environment
env = LanguageTranslationEnv()

# Load models
ppo_model = PPO.load(os.path.join("models", "ppo_language_translation.zip"), env=env)
dqn_model = DQN.load(os.path.join("models", "dqn_language_translation.zip"), env=env)

# Helper function for logging actions
def record_actions(model, env, episodes=5):
    for episode in range(episodes):
        obs = env.reset()
        done = False
        actions_taken = []

        while not done:
            action, _ = model.predict(obs)
            actions_taken.append(action)
            obs, reward, done, _, info = env.step(action)

            # Optionally render environment to visualize
            env.render()

        print(f"Episode {episode + 1} Actions: {actions_taken}")

# Evaluate PPO model
print("Testing PPO Model")
record_actions(ppo_model, env)

# Evaluate DQN model
print("\nTesting DQN Model")
record_actions(dqn_model, env)
