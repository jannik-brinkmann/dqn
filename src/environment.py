from collections import deque

import cv2
import gym
import numpy as np
import random


class DQNEnvironment(gym.Wrapper):

    def __init__(self, environment, config):
        super().__init__(environment)
        self.config = config

        # set environment parameters
        self.lives = self.env.unwrapped.ale.lives()
        self.start_new_game = True

    def reset(self, seed):
        """
        starts an episode after loss-of-life or loss-of-game
        """
        # call Gym environment reset function only when a new game should be started
        if self.start_new_game:
            self.env.reset(seed=seed)
            self.lives = self.env.unwrapped.ale.lives()
            self.start_new_game = False

        # otherwise, advance from terminal state using no-op action
        else:
            self.env.step(self.env.unwrapped.get_action_meanings().index('NOOP'))

        # some environments remain unchanged until 'FIRE' action is performed
        if 'FIRE' in self.env.unwrapped.get_action_meanings():
            self.env.step(self.env.unwrapped.get_action_meanings().index('FIRE'))

        # execute a random number of 'NOOP' actions to randomize the initial game state
        n_wait_actions = random.randint(1, self.config.max_n_wait_actions)
        for _ in range(n_wait_actions):
            self.env.step(self.env.unwrapped.get_action_meanings().index('NOOP'))

    def step(self, action):
        """
        executes action with frame-skipping and considers end-of-life as end-of-episode
        """
        # initialize return parameters
        step_observation = deque(maxlen=2)
        step_reward = 0
        step_done = False
        step_info = {}

        # repeat selected action over k frames to enable faster simulation (frame-skipping technique)
        for _ in range(self.config.action_repeat):

            # execute a single step using Gym environment step function
            observation, reward, done, info = self.env.step(action)

            # update return parameters
            step_observation.append(observation)
            step_reward = step_reward + reward
            step_done = step_done or done
            step_info = info

            # consider end-of-life as end-of-episode to improve value estimation
            if self.env.unwrapped.ale.lives() < self.lives:
                self.lives = self.env.unwrapped.ale.lives()
                step_done = True

            # set indication that the game is over, i.e. no lives left
            if self.env.unwrapped.ale.lives() == 0:
                self.start_new_game = True

            if step_done:
                break

        # remove flickering and reduce observation dimensionality over last two frames
        step_observation = self._observation_preprocessing(step_observation[0], step_observation[-1])
        return step_observation, step_reward, step_done, step_info

    def _observation_preprocessing(self, frame_a, frame_b):
        # select maximum value for each pixel colour over previous and current frame to remove flickering
        observation = np.maximum(frame_a, frame_b, dtype=np.float32)

        # reduce observation dimensionality through gray-scaling and down-sampling to 84 x 84
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, (84, 84), interpolation=cv2.INTER_LINEAR)
        return observation


