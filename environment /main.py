import sys
import os
import cv2
import time
import numpy as np
from stable_baselines3 import PPO
from environment.tourism_env import TourismEnv  # Import your custom environment

# Ensure correct module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load the trained model
MODEL_PATH = "models/trained_agent.zip"  # Update if needed
model = PPO.load(MODEL_PATH)

# Video recording function
def record_video_simulation(num_episodes=5):
    env = TourismEnv()  # Initialize environment
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec

    for episode in range(num_episodes):
        print(f"Starting episode {episode + 1}...")
        done = False
        obs, _ = env.reset()  # Ensure correct unpacking for Gym v26+
        video_filename = f'agent_simulation_episode_{episode + 1}.avi'
        out = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))  # Ensure correct resolution

        while not done:
            action, _ = model.predict(obs)  # Get action from trained model
            obs, reward, done, truncated, info = env.step(action)
            
            frame = env.render()  # Capture frame
            
            if frame is not None and frame.shape[:2] == (480, 640):
                out.write(frame)  # Write frame to video file
                cv2.imshow(f'Agent Simulation - Episode {episode + 1}', frame)
            else:
                print(f"Invalid frame at episode {episode + 1}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit if 'q' is pressed

        out.release()  # Close video file
        print(f"Episode {episode + 1} recorded as {video_filename}")

    env.close()  # Ensure environment closes
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Run the simulation
if __name__ == "__main__":
    record_video_simulation(num_episodes=5)
