from collections import deque

import cv2
import gym
import numpy as np
import random


class DQNEnvironment(gym.Wrapper):

    def __init__(self, environment, config):
        super().__init__(environment)

        assert 'NOOP' in self.env.unwrapped.get_action_meanings()
        self.noop_action = self.env.unwrapped.get_action_meanings().index('NOOP')

        # max. number of 'do nothing' actions at the beginning of a new game
        self.max_n_wait_actions = config.max_n_wait_actions

        # number of times an action should be repeated in each step (frame-skipping)
        self.action_repeat = config.action_repeat

        # consider end-of-life as end-of-episode, but only call Gym environment reset function if all lives exhausted
        self.lives = 0
        self.start_new_game = True

        self._observation_buffer = deque(maxlen=2)

    def start_episode(self, seed):

        if self.start_new_game:
            observation = self.env.reset(seed=seed)
        else:
            # advance using a 'do nothing' action
            observation, _, _, _ = self.env.step(self.noop_action)
        self.lives = self.env.unwrapped.ale.lives()

        # some environments remain unchanged until impulse action is performed
        if 'FIRE' in self.env.unwrapped.get_action_meanings():
            observation, _, _, _ = self.env.step(self.env.unwrapped.get_action_meanings().index('FIRE'))

        return observation

    def start_random_episode(self, seed):

        observation = self.start_episode(seed=seed)

        # execute a random number of 'do nothing' actions to randomize the initial game state
        n_wait_actions = random.randint(1, self.max_n_wait_actions + 1)
        for _ in range(n_wait_actions):
            observation, _, episode_done, _ = self.env.step(self.noop_action)

            # in case episode ends during random actions, call Gym environment reset function
            if episode_done:
                self.env.reset()

        return observation

    def step(self, action):

        assert self.action_space.contains(action)

        cumulative_reward = 0
        done = False

        for _ in range(self.action_repeat):

            observation, reward, done, info = self.env.step(action)

            self._observation_buffer.append(observation)
            cumulative_reward = cumulative_reward + reward
            self.start_new_game = done

            # consider end-of-life as end-of-episode to improve value estimation
            if self.lives > self.env.unwrapped.ale.lives():
                done = True
            self.lives = self.env.unwrapped.ale.lives()

            if done:
                break

        # select maximum value for each pixel colour over the last two frame to remove flickering
        reduced_observation = np.maximum(self._observation_buffer[0], self._observation_buffer[-1], dtype=np.float32)

        # reduce observation dimensionality through gray-scaling and down-sampling to 84 x 84
        reduced_observation = cv2.cvtColor(reduced_observation, cv2.COLOR_BGR2GRAY)
        reduced_observation = cv2.resize(reduced_observation, (84, 84), interpolation=cv2.INTER_LINEAR)
        return reduced_observation, cumulative_reward, done, info
