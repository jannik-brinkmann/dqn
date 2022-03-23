from collections import deque

import cv2
import gym
import numpy as np


class DQNEnvironment(gym.Wrapper):

    def __init__(self, environment, config):
        super().__init__(environment)
        self.config = config

        self.environment = gym.make(environment)
        self.action_space = self.environment.action_space
        self.observation_space = self.environment.observation_space

    def reset(self, seed):
        return self.environment.reset(seed=seed)

    def step(self, action):
        step_observation = deque(maxlen=2)
        step_reward = 0
        step_done = False
        step_info = {}

        # repeat selected action over k frames to enable faster simulation (frame-skipping technique)
        for _ in range(self.config.action_repeat):
            observation, reward, done, info = self.environment.step(action)

            step_observation.append(observation)
            step_reward = step_reward + reward
            step_done = step_done or done
            step_info = info

        # select maximum value for each pixel colour over previous and current frame to remove flickering
        step_observation = np.maximum(step_observation[1], step_observation[-1], dtype=np.float32)

        # reduce observation dimensionality through gray-scaling and down-sampling to 84 x 84
        step_observation = cv2.cvtColor(step_observation, cv2.COLOR_BGR2GRAY)
        step_observation = cv2.resize(step_observation, (84, 84), interpolation=cv2.INTER_LINEAR)
        return step_observation, step_reward, step_done, step_info
