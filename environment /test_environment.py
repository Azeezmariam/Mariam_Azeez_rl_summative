import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.language_translation_env import LanguageTranslationEnv

# Create the environment
env = LanguageTranslationEnv()

# Reset the environment and get the initial state
state = env.reset()
print(f"Initial State: {state}")

# Take some random actions and observe the results
for _ in range(3):
    action = env.action_space.sample()  # Random action
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    env.render()
