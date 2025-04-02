from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class EnvironmentVisualizer:
    def __init__(self, env):
        self.env = env

    def draw_grid(self):
        glColor3f(0.0, 0.0, 0.0)  # Grid color (black)
        glBegin(GL_LINES)
        for i in range(6):  # Drawing grid lines (adjust depending on grid size)
            glVertex2f(i, 0)
            glVertex2f(i, 5)  # 5 is the grid size (5x5)
            glVertex2f(0, i)
            glVertex2f(5, i)
        glEnd()

    def draw_agent(self, state):
        glColor3f(0.0, 1.0, 0.0)  # Green color for the agent
        agent_x, agent_y = state, state  # Assuming state is an integer for simplicity

        # Debugging output
        print(f"Drawing agent at ({agent_x}, {agent_y})")

        glPushMatrix()
        glTranslatef(agent_x, agent_y, 0)  # Move agent to its position based on state
        glutSolidCube(1.0)  # Increase size of the agent (cube size 1.0 instead of 0.5)
        glPopMatrix()

    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        self.draw_grid()

        # Ensure the state is valid
        if hasattr(self.env, 'state'):
            self.draw_agent(self.env.state)  # Pass the agent's state for visualization
        else:
            print("Environment does not have a 'state' attribute")

        glutSwapBuffers()

    def start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(500, 500)  # Set a reasonable window size
        glutCreateWindow("Environment Visualization")
        glClearColor(0.8, 0.8, 0.8, 1.0)  # Set background color (light gray)
        glEnable(GL_DEPTH_TEST)
        
        # Set orthogonal projection for a 2D grid, ensuring we cover the area where the agent moves
        glOrtho(0, 5, 0, 5, -1, 1)  # Adjust the view if needed (x, y ranges)

        # Print the initial state to debug
        print(f"Initial state: {self.env.state}")

        glutDisplayFunc(self.display)
        glutMainLoop()

# Example usage:
if __name__ == "__main__":
    # Assuming 'env' is your environment object
    from environment.language_translation_env import LanguageTranslationEnv  # Replace with your actual env
    env = LanguageTranslationEnv()

    # Ensure the environment state is initialized
    state = env.reset()  # This should initialize the state
    
    # Print the state to debug
    print("Initial state:", state)

    # Set the state explicitly if necessary (for debugging purposes)
    if isinstance(state, int):  # Adjusted for simplicity
        env.state = state
    else:
        print("State format issue, ensure the environment has a valid state attribute.")

    visualizer = EnvironmentVisualizer(env)
    visualizer.start()
