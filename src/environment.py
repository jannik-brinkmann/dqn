import gym

import numpy as np
from collections import deque
from src.utils import preprocessing


class DQNEnvironment(gym.Wrapper):

    def __init__(self, environment, config):
        super().__init__(environment)
        self.config = config

        self.environment = gym.make(environment)
        self.action_space = self.environment.action_space
        self.observation_space = self.environment.observation_space

    def reset(self):
        return self.environment.reset()

    def step(self, action):

        # initialize return parameters
        step_observation = deque(maxlen=2)
        step_reward = 0
        step_done = False
        step_info = {}

        for _ in range(self.config.action_repeat):  # frame-skipping technique
            observation, reward, done, info = self.environment.step(action)

            step_observation.append(observation)
            step_reward = step_reward + reward
            step_done = step_done or done
            step_info = info

        step_observation = preprocessing(step_observation[1], step_observation[-1])
        # step_reward = np.clip(step_reward, -1, 1)  # clip reward
        return step_observation, step_reward, step_done, step_info
