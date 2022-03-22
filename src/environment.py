import gym

import numpy as np


class DQNEnvironment(gym.Wrapper):

    def __init__(self, environment):
        super().__init__(environment)

        self.environment = gym.make(environment)
        self.action_space = self.environment.action_space
        self.observation_space = self.environment.observation_space

    def reset(self):
        return self.environment.reset()

    def step(self, action):
        observation, reward, done, info = self.environment.step(action)
        reward = np.clip(reward, -1, 1)
        return observation, reward, done, info
