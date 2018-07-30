import cv2
import numpy as np
from collections import deque

class GymEnvironment(object):
    def __init__(self, gym_env, width, height, agent_history_length):
        self.env = gym_env
        self.width = width
        self.height = height
        self.agent_history_length = agent_history_length

        self.gym_actions = range(gym_env.action_space.n)
        if gym_env.spec.id == "Breakout-v0":
            self.gym_actions = [1, 2, 3]
        self.state_buffer = deque()

    def _convert2gray_frame(self, observation):
        return cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (self.width, self.height))

    def get_initial_state(self):
        self.state_buffer = deque()
        x_t = self.env.reset()
        x_t = self._convert2gray_frame(x_t)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis = 0)
        for i in range(self.agent_history_length-1):
            self.state_buffer.append(x_t)
        return s_t

    def step(self, action_index):
        x_t1, r_t, terminal, info = self.env.step(self.gym_actions[action_index])
        x_t1 = self._convert2gray_frame(x_t1)
        previous_frames = np.array(self.state_buffer)
        s_t1 = np.empty((self.agent_history_length, self.height, self.width))
        s_t1[:self.agent_history_length-1, ...] = previous_frames
        s_t1[self.agent_history_length-1] = x_t1
        self.state_buffer.popleft()
        self.state_buffer.append(x_t1)
        return s_t1, r_t, terminal, info

    def render(self):
        self.env.render()
