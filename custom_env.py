import gym
import numpy as np
import pygame

class TourismEnv(gym.Env):
    def __init__(self):
        super(TourismEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=10, shape=(3,), dtype=np.float32)
        self.state = np.array([4.0, 3.0, 7.0])

        # Initialize Pygame
        self.screen_width = 640
        self.screen_height = 480
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        
    def reset(self):
        self.state = np.array([4.0, 3.0, 7.0])
        return self.state

    def step(self, action):
        self.state = np.array([4.0, 3.0, 7.0])  # Example new state
        reward = 1.0
        done = False
        truncated = False
        info = {}
        return self.state, reward, done, truncated, info

    def render(self):
        self.screen.fill((255, 255, 255))  # White background

        # Example: Draw an agent as a circle
        agent_x = int(self.state[0] * 50) % self.screen_width  # Scale and wrap around
        agent_y = int(self.state[1] * 50) % self.screen_height
        pygame.draw.circle(self.screen, (0, 0, 255), (agent_x, agent_y), 20)  # Blue agent
        
        pygame.display.flip()
        self.clock.tick(30)  # Limit to 30 FPS
        
        # Convert Pygame screen to NumPy array
        frame = np.array(pygame.surfarray.array3d(self.screen))
        frame = np.transpose(frame, (1, 0, 2))  # Convert to (height, width, 3)
        
        return frame  # This frame will be recorded
