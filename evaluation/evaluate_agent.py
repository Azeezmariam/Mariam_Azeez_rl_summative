import os
import gymnasium as gym
import imageio
from stable_baselines3 import DQN, PPO
import numpy as np
from gym.envs.registration import register

# Register your environment
register(
    id='LanguageTranslation-v0',  
    entry_point='environment.language_translation_env:LanguageTranslationEnv',  
)

# Initialize your environment
env = gym.make('LanguageTranslation-v0', render_mode='rgb_array')

# Your trained model path (adjust according to your model type and saved path)
MODEL_TYPE = "dqn"  # Choose between 'dqn' or 'ppo'
model_dir = "models/dqn" if MODEL_TYPE == "dqn" else "models/pg"
model_path = os.path.join(model_dir, f"{MODEL_TYPE}_language_translation.zip")

# Load the trained model (DQN or PPO)
if MODEL_TYPE == "dqn":
    model = DQN.load(model_path, env=env)
elif MODEL_TYPE == "ppo":
    model = PPO.load(model_path, env=env)
else:
    raise ValueError("Invalid model type. Choose 'dqn' or 'ppo'.")

# Set number of episodes to visualize
EPISODES = 5
video_path = os.path.join("evaluation", f"{MODEL_TYPE}_evaluation.mp4")

# Create a list to store frames for the video
frames = []

# Evaluate the model and record the video
for episode in range(EPISODES):
    obs, _ = env.reset()  # Reset the environment and get the initial observation
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Predict the action
        obs, reward, done, truncated, info = env.step(action)  # Take the step
        episode_reward += reward

        # Capture the frame for the video
        frame = env.render()
        frames.append(frame)

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

# Save the captured frames as a video
imageio.mimsave(video_path, frames, fps=10)
print(f"Video saved at: {video_path}")

# Close the environment
env.close()
