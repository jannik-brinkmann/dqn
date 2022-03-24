import argparse
import itertools
import os
from collections import deque
import logging

import numpy as np

from src.agent import DQNAgent
from src.environment import DQNEnvironment


def deep_q_learning(environment, config):

    environment = DQNEnvironment(environment=environment, config=config)
    agent = DQNAgent(action_space=environment.action_space, config=config)

    # training parameters
    episode_reward_buffer = deque(maxlen=100)

    n_steps = 1
    for episode in itertools.count():

        # set environment to initial screen and clear agent's observation buffer
        environment.reset(seed=config.seed)
        agent.clear_observations()

        # set episode parameters
        episode_done = False
        episode_reward = 0.0

        while not episode_done:

            # select action using a uniform random policy to populate the replay memory D
            if not agent.memory.is_full():
                action = agent.action_space.sample()

            # otherwise, select action using an epsilon-greedy strategy
            else:
                action = agent.select_action(n_steps=n_steps)
                n_steps = n_steps + 1

            # execute action a_t in emulator
            observation, step_reward, episode_done, _ = environment.step(action)

            # observe image x_(t+1); store transition in replay memory D
            agent.append_observation(action, observation, step_reward, episode_done)

            # update the action-value function Q every k-th action
            if n_steps % config.update_frequency == 0:
                agent.update_network()

            # update target network every k-th parameter upgrade
            if n_steps % (config.update_frequency * config.target_network_update_frequency) == 0:
                agent.update_target_network()

            episode_reward += step_reward

        episode_reward_buffer.append(episode_reward)

        print(f'Episode: {episode}, Step: {n_steps}, Reward: {episode_reward}, Avg. Reward: {np.mean(episode_reward_buffer):.2f}')

        # store network weights every k-th episode
        if episode % config.weight_save_frequency == 0:
            agent.save_model_weights(os.path.join(os.path.dirname(__file__), os.pardir, 'weights', 'checkpoint.pth.tar'))