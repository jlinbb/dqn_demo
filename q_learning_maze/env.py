import time
import sys
import numpy as np

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

WIDTH = 4
HEIGHT = 3
UNIT = 40


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('MAZE')
        self.geometry('{0}x{1}'.format(WIDTH*UNIT, HEIGHT*UNIT))
        self._build_maze()

    def _create_object(self, center_x, center_y, size, shape='oval', color='yellow'):
        """create different object of maze including robot, bomb and treasure
        """
        if(shape.lower() == 'oval'):
            object = self.canvas.create_oval(
                center_x - size, center_y - size,
                center_x + size, center_y + size,
                fill=color
            )
        elif(shape.lower() == 'rectangle'):
            object = self.canvas.create_rectangle(
                center_x - size, center_y - size,
                center_x + size, center_y + size,
                fill=color
            )
        return object


    def _build_maze(self):
        """draw maze including the whole map and different objects
        """
        self.canvas = tk.Canvas(self, bg='white', width=WIDTH*UNIT, height=HEIGHT*UNIT)

        for c in range(0, WIDTH * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0 , c , HEIGHT * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        self.origin = np.array([20, 20])  # center
        self.robot_center = self.origin + np.array([0, UNIT*2])
        self.robot_size = 15
        self.robot = self._create_object(
            self.robot_center[0], self.robot_center[1], self.robot_size,
            shape='oval', color='yellow'
        )

        bomb1_center = self.origin + UNIT
        bomb_size = 15
        self.bomb1 = self._create_object(
            bomb1_center[0], bomb1_center[1], bomb_size,
            shape='rectangle', color='red'
        )
        bomb2_center = self.origin + np.array([UNIT * 3, UNIT])
        self.bomb2 = self._create_object(
            bomb2_center[0], bomb2_center[1], bomb_size,
            shape='rectangle', color='red'
        )

        treasure_center = self.origin + np.array([UNIT * 3, 0])
        treasure_size = 15
        self.treasure = self._create_object(
            treasure_center[0], treasure_center[1], treasure_size,
            shape='rectangle', color='green'
        )
        self.canvas.pack()
        # self.canvas.wait_window() # preview maze


    def reset(self):
        """reset the game, init the coords of robot
        """
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.robot)
        self.robot = self._create_object(
            self.robot_center[0], self.robot_center[1], self.robot_size,
            shape='oval', color='yellow'
        )
        return self.canvas.coords(self.robot)

    def step(self, action):
        """operation of the robots and return the coords of robo, reward and  final state
        """
        s = self.canvas.coords(self.robot)
        base_action = np.array([0, 0])
        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT  # up
        elif action == 1:
            if s[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT  # down
        elif action == 2:
            if s[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT  # right
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT  # left

        self.canvas.move(self.robot, base_action[0], base_action[1])
        s = self.canvas.coords(self.robot)  # next coords

        if s == self.canvas.coords(self.treasure):
            reward = 1
            done = True
            s = 'terminal'
            print('Mission complete')
        elif s == self.canvas.coords(self.bomb1) or s == self.canvas.coords(self.bomb2):
            reward = -1
            done = True
            s = 'terminal'
            print('boom! failed!')
        else:
            reward = 0
            done = False
        return s, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()




