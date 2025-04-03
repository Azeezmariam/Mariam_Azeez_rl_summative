import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np
import cv2
import time

class TourismEnvVisualization:
    def __init__(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL)
        gluPerspective(45, (self.width / self.height), 0.1, 50.0)
        glTranslatef(0.0, 0.0, -5)

        # Video recording setup
        self.recording = True
        self.frame_rate = 30
        self.video_filename = "agent_simulation.avi"
        self.out = cv2.VideoWriter(self.video_filename, cv2.VideoWriter_fourcc(*'XVID'), self.frame_rate, (self.width, self.height))

    def draw_agent(self, x, y, z):
        glPushMatrix()
        glTranslatef(x, y, z)
        glColor3f(0.0, 0.0, 1.0)  # Blue color
        glutSolidSphere(0.2, 20, 20)  # Draw sphere as agent
        glPopMatrix()

    def draw_environment(self):
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_QUADS)
        for x in range(-5, 6):
            for y in range(-5, 6):
                glVertex3f(x, y, -0.1)
                glVertex3f(x + 1, y, -0.1)
                glVertex3f(x + 1, y + 1, -0.1)
                glVertex3f(x, y + 1, -0.1)
        glEnd()

    def capture_frame(self):
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
        image = np.flip(image, axis=0)  # Flip vertically
        return image

    def run_simulation(self, steps=100):
        running = True
        angle = 0
        x, y, z = 0, 0, 0

        for _ in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glPushMatrix()
            glRotatef(angle, 0, 1, 0)
            self.draw_environment()
            self.draw_agent(x, y, z)
            glPopMatrix()
            pygame.display.flip()

            frame = self.capture_frame()
            self.out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            x = np.sin(time.time()) * 2  # Simulating movement
            y = np.cos(time.time()) * 2
            angle += 1  # Rotate scene for better visualization

        self.out.release()
        pygame.quit()

if __name__ == "__main__":
    vis = TourismEnvVisualization()
    vis.run_simulation()
